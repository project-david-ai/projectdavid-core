```

docker run -d `
  --gpus=all `
  -v ollama:/root/.ollama `
  -p 11434:11434 `
  --name ollama `
  ollama/ollama
```

```
curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:4b",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": false
}'


curl http://localhost:11434/api/chat -d '{
  "model": "qwen3:4b",
  "messages": [{
    "role": "user",
    "content": "Why is the sky blue?"
  }],
  "stream": true
}'


```
