# Contributing

## Install
```
pip install -e ".[develop]"

pre-commit install
```

## Run CI locally
To run the CI locally:

### [optional] Start the cache.

Download
https://github.com/sp-ricard-valverde/github-act-cache-server

Run
```
ACT_CACHE_AUTH_KEY=foo docker compose up --build
```

Also modify your .actrc.


### Run


Setup (make sure docker is installed):
```
brew install act
```

Run act
```
act -j develop
```


## Testing

To run the tests:

```
pytest
```

To run all the tests, including long ones

```
pytest -m "long"
```
