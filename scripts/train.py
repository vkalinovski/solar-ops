from __future__ import annotations

import json

from solarflare_app.modeling.trainer import HorizonTrainer
from solarflare_app.settings import TrainingSettings


def main() -> None:
    settings = TrainingSettings()
    trainer = HorizonTrainer(settings=settings, bundle_dir=settings.bundle_dir)
    results = trainer.train_all()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
