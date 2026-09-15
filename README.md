# Sample AIWolf Python agent using aiwolf package
Sample AIWolf agent written in Python using [aiwolf package](https://github.com/AIWolfSharp/aiwolf-python).

## Explainable role estimation

For the standard five-player role set (2 villagers, 1 seer, 1 possessed,
and 1 werewolf), `SampleVillager` now uses an exact possible-world role
estimator. Public statements are weighted evidence rather than accepted facts.
The estimator retains a structured argument made from observations, named
rules, and competing hypotheses; probability is reported only as a summary.
Vote selection is deterministic for identical game histories.

The design rationale and requirements are recorded in `docs/`. Note that the
current repository `.gitignore` excludes that directory.

## Prerequisites
* Python 3.9
* [aiwolf package]((https://github.com/AIWolfSharp/aiwolf-python)). 
You can install aiwolf package as follows,
```
pip install git+https://github.com/AIWolfSharp/aiwolf-python.git
```
## How to use
Suppose the AIWolf server at localhost is waiting a connection from an agent on port 10000.
You can connect this sample agent to the server as follows,
```
python start.py -h locahost -p 10000 -n name_you_like
```

## Tests

```console
python -m unittest discover -s tests -v
```
