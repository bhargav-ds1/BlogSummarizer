start: ## start the docker container
	@echo "Starting docker container"
	@docker compose up
	@echo "Container started - Streamlit-app:http://localhost:8501 \n Observability-Phoenix:http://localhost:6006"

start-test: ## Run a test and gather evaluation metrics
	@docker compose exec blog_summarizer sh -c "deepeval login --api-key $DEEPEVAL_API_KEY"
	@docker compose exec blog_summarizer sh -c "deepeval test run Tests/test_blog_summarizer.py"