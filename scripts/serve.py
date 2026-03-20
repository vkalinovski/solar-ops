from __future__ import annotations

import uvicorn

from solarflare_app.settings import RuntimeSettings


def main() -> None:
    settings = RuntimeSettings()
    uvicorn.run(
        "solarflare_app.api.app:create_app",
        host=settings.host,
        port=settings.port,
        reload=False,
        factory=True,
    )


if __name__ == "__main__":
    main()
