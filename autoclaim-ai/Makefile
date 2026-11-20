.PHONY: pull-models up down test-backend dev-backend dev-frontend

pull-models:
	ollama pull qwen2.5vl:7b
	ollama pull qwen2.5:7b
	ollama pull nomic-embed-text

up:
	docker compose up --build -d

down:
	docker compose down

test-backend:
	cd backend && pytest -v

dev-backend:
	cd backend && uvicorn app.main:app --reload --port 8000

dev-frontend:
	cd frontend && npm install && npm run dev
