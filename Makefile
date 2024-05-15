start-app: ## start the streamlit app
	@echo "Startign streamlit app based on the Configurations from config.py file"
	@streamlit run Apps/Streamlit_app/app.py
	@echo "Application started - Streamlit-app: http://localhost:8501 \n Observability-Phoenix: http://localhost:6006"

start-test: ## run tests
	@echo "Running tests from Tests package"
	# By logging in with a confident ai account, we can visualize the tests with UI.
	#@deepeval login
	# You have to create an account here https://app.confident-ai.com/auth/signup which is a free-trail for 7 days.
	# Obtain an API-Key in the project-details tab and provide it when prompted during login
	@deepeval test run Tests/test_blog_summarizer.py

start-docker: ## start the docker container
	@echo "Starting docker container"
	@docker compose up
	@echo "Container started - Streamlit-app:http://localhost:8501 \n Observability-Phoenix:http://localhost:6006"

