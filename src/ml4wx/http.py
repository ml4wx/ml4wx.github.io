"""Shared HTTP client: polite rate limiting and exponential backoff on 429 / 5xx."""

from __future__ import annotations

import logging
import threading
import time

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "ml4wx/0.1 (+https://github.com/ml4wx/ml4wx.github.io)"


class BudgetExceeded(RuntimeError):
    """The API's daily budget is used up (OpenAlex); retrying today will not help."""


class Client:
    def __init__(self, min_interval: float = 0.0, max_retries: int = 6, timeout: float = 60.0):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last = 0.0
        self._lock = threading.Lock()
        self._http = httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )

    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def request(self, method: str, url: str, **kw) -> httpx.Response:
        """Returns the response for 2xx/3xx and 404; retries 429 and 5xx with backoff."""
        delay = max(self.min_interval, 2.0)
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self._http.request(method, url, **kw)
            except httpx.TransportError as e:
                log.warning("%s %s failed (%s), retry %d", method, url, e, attempt + 1)
            else:
                if r.status_code < 400 or r.status_code == 404:
                    return r
                if r.status_code != 429 and r.status_code < 500:
                    r.raise_for_status()
                left = r.headers.get("x-ratelimit-remaining-usd")
                if r.status_code == 429 and left is not None and float(left) <= 0:
                    raise BudgetExceeded(f"{url}: daily budget used up")
                log.warning("%s %s -> %d, retry %d in %.0fs", method, url, r.status_code, attempt + 1, delay)
            if attempt == self.max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 120)
        raise RuntimeError(f"{method} {url} failed after {self.max_retries} retries")

    def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        return self.request("GET", url, params=params, headers=headers)

    def post(self, url: str, json: dict, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        return self.request("POST", url, json=json, params=params, headers=headers)
