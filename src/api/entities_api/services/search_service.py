import re
import time

import requests
from projectdavid_common.utilities.logging_service import LoggingUtility

from src.api.entities_api.constants.platform import WEB_SEARCH_BASE_URL
from src.api.entities_api.utils import count_tokens

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

    def __init__(self, firecrawl_url="http://localhost:3002/v1/crawl"):
        """
        Initialize the FirecrawlService with the Firecrawl API URL.
        Set retry parameters for job completion checks.
        """
        self.firecrawl_url = firecrawl_url
        self.max_retries = 10
        self.retry_delay = 2
        self.token_count = []

    def crawl_url(self, url: str) -> str:
        """
        Submit a crawl request for a specific URL and return the job ID.
        Sends a POST request to the Firecrawl API.
        """
        logging_utility.info(f"Submitting crawl request for URL: {url}")
        response = requests.post(self.firecrawl_url, json={"url": url}, timeout=30)

        if response.status_code == 200:
            job_id = response.json().get("id")
            logging_utility.info(f"Crawl job submitted successfully. Job ID: {job_id}")
            return job_id
        logging_utility.error(
            f"Failed to submit crawl request. Status code: {response.status_code}"
        )
        return None

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

    def search_orchestrator(self, query, max_pages):
        for i in range(max_pages):
            url_to_crawl = f"{WEB_SEARCH_BASE_URL}{query}&page={i + 1}"
            job_id = self.crawl_url(url_to_crawl)
            if job_id:
                results = self.wait_for_completion(job_id)
                if results:
                    logging_utility.info("Crawl results retrieved successfully.")
                    print(results)
                    results_data = results["data"]
                    print(results_data)
                    results_markdown_dict = results_data[0]
                    print(results_markdown_dict)
                    current_page = extract_skip_to_content_url(
                        results_markdown_dict["markdown"]
                    )
                    if current_page:
                        tokens_per_current_page = count_tokens(
                            input_string=current_page
                        )
                        self.token_count.append(tokens_per_current_page)
                        print(self.token_count)
                        print(current_page)
                    time.sleep(0.1)


if __name__ == "__main__":
    service = FirecrawlService()
    query = "war"
    service.search_orchestrator(query=query, max_pages=50)
