import hashlib
import hmac
import io
import os
import socket
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from fastapi import HTTPException
from smb.SMBConnection import SMBConnection

from src.api.entities_api.models.models import File


class SambaClient:

    def __init__(
        self,
        server: str,
        share: str,
        username: str,
        password: str,
        domain: Optional[str] = None,
        port: int = 445,
        max_retries: int = 5,
        retry_delay: int = 2,
    ):
        self.server = server
        self.share = share
        self.username = username
        self.password = password
        self.domain = domain or "WORKGROUP"
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.client_name = socket.gethostname()
        self.server_name = server
        self.conn = None
        for retry in range(self.max_retries):
            try:
                print(
                    f"Attempting to connect to SMB server (attempt {retry + 1}/{self.max_retries})..."
                )
                self._connect()
                print(f"Successfully connected to SMB server on attempt {retry + 1}")
                break
            except Exception as e:
                print(f"Connection attempt {retry + 1} failed: {str(e)}")
                if retry < self.max_retries - 1:
                    print(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(
                        f"Failed to connect to SMB server after {self.max_retries} attempts. Last error: {str(e)}"
                    )

    def _connect(self):
        connection_params: List[Dict[str, Any]] = [
            {"use_ntlm_v2": True, "is_direct_tcp": True},
            {"use_ntlm_v2": True, "is_direct_tcp": False},
            {"use_ntlm_v2": False, "is_direct_tcp": True},
            {"use_ntlm_v2": False, "is_direct_tcp": False},
        ]
        errors = []
        for params in connection_params:
            try:
                print(f"Trying connection with parameters: {params}")
                self.conn = SMBConnection(
                    self.username,
                    self.password,
                    self.client_name,
                    self.server_name,
                    domain=self.domain,
                    **params,
                )
                try:
                    ip_address = socket.gethostbyname(self.server)
                    print(f"Resolved {self.server} to IP: {ip_address}")
                except socket.gaierror:
                    print(f"Warning: Could not resolve hostname {self.server}")
                if self.conn.connect(self.server, self.port):
                    print(f"Connected successfully with params: {params}")
                    return
                else:
                    error_msg = "Connection returned false"
                    print(f"{error_msg} with params: {params}")
                    errors.append(error_msg)
            except Exception as e:
                error_msg = str(e)
                print(f"Connection attempt failed with params {params}: {error_msg}")
                errors.append(error_msg)
        raise Exception(f"Failed to connect to SMB server. Errors: {', '.join(errors)}")

    def list_files(self, remote_dir: str = ""):
        try:
            files = self.conn.listPath(self.share, remote_dir)
            return [f.filename for f in files if f.filename not in [".", ".."]]
        except Exception as e:
            raise Exception(f"Failed to list files: {str(e)}")

    listdir = list_files

    def upload_file(self, local_path: str, remote_path: Optional[str] = None):
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")
        remote_path = remote_path or os.path.basename(local_path)
        try:
            with open(local_path, "rb") as file_obj:
                self.conn.storeFile(self.share, remote_path, file_obj)
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    def download_file(self, remote_path: str, local_path: Optional[str] = None):
        local_path = local_path or os.path.basename(remote_path)
        try:
            with open(local_path, "wb") as file_obj:
                self.conn.retrieveFile(self.share, remote_path, file_obj)
        except Exception as e:
            raise Exception(f"Failed to download file: {str(e)}")

    def download_file_to_bytes(self, remote_path: str) -> bytes:
        file_obj = io.BytesIO()
        try:
            self.conn.retrieveFile(self.share, remote_path, file_obj)
            file_obj.seek(0)
            return file_obj.read()
        except Exception as e:
            raise Exception(f"Failed to download file to bytes: {str(e)}")

    def delete_file(self, remote_path: str):
        try:
            self.conn.deleteFiles(self.share, remote_path)
        except Exception as e:
            raise Exception(f"Failed to delete file: {str(e)}")

    def create_directory(self, remote_dir: str):
        try:
            self.conn.createDirectory(self.share, remote_dir)
        except Exception as e:
            raise Exception(f"Failed to create directory: {str(e)}")

    def delete_directory(self, remote_dir: str):
        try:
            self.conn.deleteDirectory(self.share, remote_dir)
        except Exception as e:
            raise Exception(f"Failed to delete directory: {str(e)}")

    def rename(self, old_remote_path: str, new_remote_path: str):
        try:
            self.conn.rename(self.share, old_remote_path, new_remote_path)
        except Exception as e:
            raise Exception(f"Failed to rename file/directory: {str(e)}")

    def get_file_as_signed_url(self, file_id: str, expires_in: int = 3600) -> str:
        file_record = self.db.query(File).filter(File.id == file_id).first()
        if not file_record:
            raise HTTPException(
                status_code=404, detail=f"File with ID {file_id} not found"
            )
        secret_key = os.getenv("SIGNED_URL_SECRET", "default_secret_key")
        expiration_time = datetime.utcnow() + timedelta(seconds=expires_in)
        expiration_timestamp = int(expiration_time.timestamp())
        data = f"{file_id}:{expiration_timestamp}"
        signature = hmac.new(
            key=secret_key.encode(), msg=data.encode(), digestmod=hashlib.sha256
        ).hexdigest()
        query_params = {
            "file_id": file_id,
            "expires": expiration_timestamp,
            "signature": signature,
        }
        base_url = os.getenv("BASE_URL", "http://localhost:9000")
        signed_url = f"{base_url}/v1/files/download?{urlencode(query_params)}"
        return signed_url

    def find_file_by_id_to_bytes(self, file_id: str, remote_dir: str = "") -> bytes:
        """
        Locate a file on the Samba share using its file id (assumes naming convention {file_id}_{filename})
        and return its contents as bytes for in-memory processing.

        Args:
            file_id (str): The unique file identifier.
            remote_dir (str): Optional remote directory to search in.

        Returns:
            bytes: The file's content as bytes.

        Raises:
            Exception: If the file cannot be found or read.
        """
        files = self.listdir(remote_dir)
        target_file = None
        for filename in files:
            if filename.startswith(f"{file_id}_"):
                target_file = filename
                break
        if not target_file:
            raise Exception(f"File with ID {file_id} not found on share.")
        try:
            file_bytes = self.download_file_to_bytes(target_file)
            return file_bytes
        except Exception as e:
            raise Exception(f"Failed to retrieve file by ID {file_id}: {str(e)}")

    def find_file_by_id_with_name(
        self, file_id: str, remote_dir: str = ""
    ) -> (str, bytes):
        """
        Locate a file on the Samba share using its file id (assuming naming convention {file_id}_{original_filename})
        and return its original filename along with its contents as bytes.

        Args:
            file_id (str): The unique file identifier.
            remote_dir (str): Optional remote directory to search in.

        Returns:
            tuple: (original_filename, bytes) where original_filename is the file's original name with extension.

        Raises:
            Exception: If the file cannot be found or read.
        """
        files = self.listdir(remote_dir)
        target_file = None
        expected_prefix = f"{file_id}_"
        for filename in files:
            if filename.startswith(expected_prefix):
                target_file = filename
                break
        if not target_file:
            raise Exception(f"File with ID {file_id} not found on share.")
        try:
            file_bytes = self.download_file_to_bytes(target_file)
            original_filename = target_file[len(expected_prefix) :]
            return (original_filename, file_bytes)
        except Exception as e:
            raise Exception(f"Failed to retrieve file by ID {file_id}: {str(e)}")

    def download_file_as_io(
        self, file_id: str, remote_dir: str = ""
    ) -> (str, io.BytesIO):
        """
        Locate a file on the Samba share using its file id (assuming naming convention {file_id}_{original_filename}),
        download its contents, and return both the original filename and an in-memory file-like object (io.BytesIO).

        Args:
            file_id (str): The unique file identifier.
            remote_dir (str): Optional remote directory to search in.

        Returns:
            tuple: (original_filename, file_io) where:
                - original_filename (str) is the file's original name with extension.
                - file_io (io.BytesIO) is the in-memory file object containing the file's bytes.

        Raises:
            Exception: If the file cannot be found or read.
        """
        original_filename, file_bytes = self.find_file_by_id_with_name(
            file_id, remote_dir
        )
        file_io = io.BytesIO(file_bytes)
        return (original_filename, file_io)

    def save_file_to_disk(
        self, file_id: str, save_dir: str, remote_dir: str = ""
    ) -> str:
        """
        Locate a file on the Samba share using its file id (assuming naming convention {file_id}_{original_filename}),
        download its contents into an in-memory BytesIO object, and then save it to disk using the original filename.

        Args:
            file_id (str): The unique file identifier.
            save_dir (str): The local directory where the file will be saved.
            remote_dir (str): Optional remote directory on the Samba share to search in.

        Returns:
            str: The full path to the saved file.

        Raises:
            Exception: If the file cannot be found or read.
        """
        original_filename, file_io = self.download_file_as_io(file_id, remote_dir)
        import os

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, original_filename)
        with open(save_path, "wb") as f:
            file_io.seek(0)
            f.write(file_io.read())
        return save_path
