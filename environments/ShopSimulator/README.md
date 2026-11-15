# ShopSimulator environment

Embedded shopping environment used by E-Commerce Agent: product search, product details, option selection, purchases, structured observations and Reward v3. The environment identity is recorded in [EMBEDDED_SOURCE.json](EMBEDDED_SOURCE.json).

The current agent entry point is [next/scripts/start_environment.sh](../../next/scripts/start_environment.sh). Configure the isolated environment interpreter using `SHOP_PYTHON`; prepare product data and its search index before starting the service. See the [execution guide](../../docs/reproduce.md) for dependencies and protocol checks.

Generated indexes, virtual environments and runtime logs are not committed. Environment and source provenance remain separate from project-specific training contributions.
