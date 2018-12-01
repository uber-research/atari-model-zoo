Install requirements:

```
pip install click
pip install matplotlib==2.0.2
pip install mpldatacursor
```

First download data to the data path specified in JSON file and then run reducer:
```
python -m process state_reduce.json --download
```

If you already downloaded data to the specified data path, simply run reducer as:
```
python -m process state_reduce.json
```

Run visualizer:
```
python -m visualize viz_state_2d.json
```