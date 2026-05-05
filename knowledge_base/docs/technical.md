# Technical Issues

## App is slow or unresponsive
First, check our status page at status.example.com. If all systems are operational: clear browser cache, disable extensions, and try a private window. Mobile: ensure the app is on the latest version.

## Sync errors
Sync errors usually mean a stale auth token. Sign out and back in to refresh. If the error persists with code SYNC-409, the workspace has a conflict that requires merging — contact support with the workspace ID.

## API rate limits
The default API limit is 60 requests/minute on free, 600/minute on Pro, and 6,000/minute on Enterprise. A 429 response includes a Retry-After header. Burst credits accumulate up to 2x the per-minute limit.

## Integrations not loading
Reauthorize the integration under Settings → Integrations. If the third-party service has changed scopes (e.g. Google, Slack), the existing token is invalidated and must be reissued.
