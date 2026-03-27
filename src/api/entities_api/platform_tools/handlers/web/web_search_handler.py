import re
import time
from urllib.parse import quote

import requests

from src.api.entities_api.constants.platform import WEB_SEARCH_BASE_URL
from src.api.entities_api.services.logging_service import LoggingUtility

logging_utility = LoggingUtility()


def extract_skip_to_content_url(markdown):
    """
    Extract the 'Skip to content' URL from markdown content.
    Uses regex to find the pattern [Skip to content](URL).
    """
    match = re.search("\\[Skip to content\\]\\((https?://[^\\s\\)]+)\\)", markdown)
    if match:
        return match.group(1)
    return None


class FirecrawlService:

    def __init__(
        self,
        firecrawl_url="http://localhost:3002/v1/crawl",
        max_retries: int = 10,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        Initialize the FirecrawlService with the Firecrawl API URL.
        Set retry parameters for job completion checks.
        """
        self.firecrawl_url = firecrawl_url
        self.max_retries = 20
        self.retry_delay = 0.5
        self.token_count = []

    def crawl_url(self, url: str) -> str:
        """
        Submit a crawl request for a specific URL and return the job ID.
        Sends a POST request to the Firecrawl API.
        """
        logging_utility.info(f"Submitting crawl request for URL: {url}")
        request_data = {"url": url}
        logging_utility.debug(f"Request data: {request_data}")
        response = requests.post(self.firecrawl_url, json=request_data, timeout=30)

        if response.status_code != 200:
            try:
                error_detail = response.json()
                logging_utility.error(
                    f"Failed to submit crawl request. Status code: {response.status_code}, Details: {error_detail}"
                )
            except Exception as e:
                logging_utility.error(
                    f"Failed to submit crawl request. Status code: {response.status_code}, Response: {response.text}"
                )
            return None
        job_id = response.json().get("id")
        logging_utility.info(f"Crawl job submitted successfully. Job ID: {job_id}")
        return job_id

    def get_results(self, job_id: str) -> dict:
        """
        Get crawl job status and results by job ID.
        Sends a GET request to the Firecrawl API.
        """
        job_url = f"{self.firecrawl_url}/{job_id}"
        try:
            logging_utility.debug(f"Fetching results for job ID: {job_id}")
            response = requests.get(job_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging_utility.error(f"Error checking job status: {e}")
            return None

    def wait_for_completion(self, job_id: str) -> dict:
        """
        Wait for the crawl job to complete and return the results.
        Polls the job status until it's completed or max retries are reached.
        """
        retries = 0
        while retries < self.max_retries:
            time.sleep(5)
            results = self.get_results(job_id)
            if results:
                if results.get("status") == "completed":
                    logging_utility.info(f"Crawl job completed. Job ID: {job_id}")
                    return results
                elif results.get("status") == "scraping":
                    logging_utility.info(
                        f"Job still in progress. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    retries += 1
                else:
                    logging_utility.warning(
                        f"Unexpected job status: {results.get('status')}"
                    )
                    break
            else:
                logging_utility.error("Failed to retrieve job status.")
                break
        logging_utility.warning(
            f"Max retries ({self.max_retries}) reached. Job may still be in progress."
        )
        return None

    def search_orchestrator(self, query, max_pages=7):
        """
        Orchestrates the search process by crawling web pages and collecting results.

        Args:
            query (str): The search query.
            max_pages (int): The maximum number of pages to search.

        Returns:
            list: A list of results in markdown format.
        """
        logging_utility.info(
            "Starting search_orchestrator with query='%s' and max_pages=%d",
            query,
            max_pages,
        )
        results_data_list = []
        for i in range(max_pages):
            try:
                encoded_query = quote(query)
                if (
                    "localhost" in WEB_SEARCH_BASE_URL
                    or "127.0.0.1" in WEB_SEARCH_BASE_URL
                ):
                    url_to_crawl = f"{WEB_SEARCH_BASE_URL}/search?q={encoded_query}&page={i + 1}&language=auto&safesearch=0&categories=general"
                else:
                    url_to_crawl = (
                        f"{WEB_SEARCH_BASE_URL}?q={encoded_query}&first={i * 10 + 1}"
                    )
                logging_utility.debug("Generated URL to crawl: %s", url_to_crawl)
                job_id = self.crawl_url(url_to_crawl)
                if job_id:
                    logging_utility.info(
                        "Job ID %s created, waiting for completion...", job_id
                    )
                    results = self.wait_for_completion(job_id)
                    if results:
                        logging_utility.info(
                            "Crawl results retrieved successfully for job ID %s.",
                            job_id,
                        )
                        results_data = results.get("data", [])
                        if results_data:
                            results_markdown_dict = results_data[0]
                            results_data_list.append(results_markdown_dict)
                            logging_utility.debug(
                                "Added results data: %s", results_markdown_dict
                            )
                        else:
                            logging_utility.warning(
                                "No data found in results for job ID %s.", job_id
                            )
                    else:
                        logging_utility.warning(
                            "No results retrieved for job ID %s.", job_id
                        )
                else:
                    logging_utility.error(
                        "Failed to create job ID for URL: %s", url_to_crawl
                    )
            except Exception as e:
                logging_utility.exception(
                    "An error occurred in search_orchestrator at iteration %d: %s",
                    i,
                    str(e),
                )
        if results_data_list:
            logging_utility.info(
                "Search completed successfully, returning %d results.",
                len(results_data_list),
            )
        else:
            logging_utility.warning("Search completed but no results were found.")
        return results_data_list if results_data_list else None


if __name__ == "__main__":
    service = FirecrawlService()
    query = "Donald Trump"
    search = service.search_orchestrator(query=query, max_pages=1)
    print(search)
