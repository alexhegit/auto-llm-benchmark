#curl http://localhost:8000/v1/models
#curl http://localhost:8000/v1/chat/completions \
#    -H "Content-Type: application/json" \
#    -d '{
#        "model": "/data/llm/Meta-Llama-3.1-8B/",
#        "messages": [
#            {"role": "system", "content": "You are a helpful assistant."},
#            {"role": "user", "content": "Who won the world series in 2020?"}
#        ]
#    }'
curl http://localhost:8000/v1/completions \
	-H "Content-Type: application/json" \
	-d '{
        "model": "/data/llm/Meta-Llama-3.1-8B/",
	"prompt": "San Francisco is a",
	"max_tokens": 7,
	"temperature": 0
}'
