# Security Policy

## Supported Versions

Security updates are provided for the latest stable release only. Make sure you are running the latest version from PyPI before reporting a vulnerability.

| Version | Supported |
|---|---|
| >= 1.96.0 | ✅ |
| < 1.96.0 | ❌ |

---

## Reporting a Vulnerability

**Please do not open a public GitHub Issue for security vulnerabilities.**

If you discover a security-related issue in the Project David SDK, report it privately. All reports are handled with high priority.

**Email:** [engineering@projectdavid.co.uk](mailto:engineering@projectdavid.co.uk)

Include:
- A summary of the issue
- A proof of concept if available
- The SDK version you are using

**Acknowledgment:** within 48 hours
**Resolution:** coordinated fix and release before public disclosure

---

## Security Notes

The SDK is a client library. It has no listening ports, no webhook handlers, and no inbound network surface. The security considerations below are relevant to operators integrating the SDK into their applications.

**Credential handling**

The API key is passed to the `Entity` constructor and held in memory for the lifetime of the client object. It is not written to disk, not logged, and not included in exception messages. Operators should treat the API key as a secret and avoid passing it through environment variables that are visible to untrusted processes.

**Transport**

All requests are sent to the `base_url` configured at initialisation. For production deployments this should be an HTTPS endpoint. Operators running the SDK against a local HTTP instance should be aware that traffic including the API key is unencrypted on the wire.

**Dependency chain**

The SDK depends on `httpx`, `pydantic`, and `projectdavid-common`. Operators in air-gapped or high-assurance environments should audit the full dependency tree before deployment. Pin dependencies in your application to prevent unexpected updates pulling in unreviewed code.

**No server-side enforcement**

The SDK does not validate API keys, enforce rate limits, or check permissions. All enforcement happens server-side. A compromised or leaked API key should be revoked immediately via the API key management endpoints on the platform.

---

## Responsible Disclosure

The Project David SDK is maintained by a solo engineer. We appreciate your patience and your help in keeping the ecosystem safe.

*Project David is created and maintained by Francis Neequaye Armah.*
*All intellectual property is solely owned by the author.*
