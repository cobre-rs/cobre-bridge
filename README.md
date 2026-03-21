# cobre-bridge

Convert power system data formats to [Cobre](https://github.com/cobre-rs/cobre) input format.

## Installation

```bash
pip install cobre-bridge
```

## Usage

```bash
cobre-bridge convert newave <SRC_DIR> <DST_DIR>
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src/
```

## License

Apache-2.0
