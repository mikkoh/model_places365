
# Places365

<a href="http://sdk.runwayml.com" target="_blank"><img src="https://runway.nyc3.cdn.digitaloceanspaces.com/assets/github/runway-badge.png" width=100/></a>

This is the repo for Places365 port into RunwayML

## Testing the Model

```bash
pip install -r requirements.txt

python runway_model.py
```

You should see an output similar to this, indicating your model is running.

```
Setting up model...
Starting model server at http://0.0.0.0:8000...
```

You can test your model once its running by POSTing a caption argument to the the `/generate` command.

```bash
curl -X POST -H "Content-Type: application/json" -d '{"image" : "'"$( base64 ./fjords.jpg)"'"}' http://localhost:8000/classify
```

You should receive a JSON object back, containing a cryptic base64 encoded URI string that represents a red image:

```
{"label": "bedroom"}
```

