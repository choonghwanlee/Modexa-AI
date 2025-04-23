# Modexa-AI

## Motivation

Modexa AI is an agentic data scientist acting as the application layer bridging enterprise data science teams with business users.

Modexa uses agentic planning and decision-making to

- break down business user's high-level questions to actionable data science tasks
- dynamically decide what data and ML models to use to answer the user's questions.

This saves enterprise data science teams months of time building dashboards or building custom software to deploy their models, and also provides end business users with a dynamic & proactive method of engaging with the company's data.

## Running the App

The app is deployed via streamlit on [this website](https://modexa-ai.streamlit.app/)

To run locally,

```bash
git clone https://github.com/choonghwanlee/Modexa-AI
pip install -r requirements.txt
streamlit run main.py
```

## Data & Models

We use the [Olist database](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce/data), a real-world e-commerce database comprised of multiple datasets. We store the database in Snowflake to simulate real-world data science environments.

With this data, we train `customer lifetime value` and `churn prediction` models using scikit-learn. The trained models are stored as **joblib** files, which can be found under `/models`.

## Agentic Approach

We use a hybrid Plan-and-Reflect + ReAct structure, where we:

1. Plan high level step-by-step plan, breaking down the user's question
2. Loop through each step:
   - Use a ReAct structure to flexibly work through each step
   - Log “think, act, observe” trace to context history
   - Save intermediary function output variables to a ‘Scratchpad’
3. Use a judge after each step finishes to determine whether to move on or not
4. Use context aggregated + variables stored to generate response

Tools we had include:

1. Predict Customer Lifetime Value
2. Predict Churn
3. Text to SQL
4. Write Python Code
5. Think/Reflect

## Evaluation

We use a LLM-as-a-judge to evaluate both the planning and response generation capabilities of the app.

We score from 1-5, evaluating our plan on three categories: conciseness, feasibility, and effectiveness

Results are shown below:

| Metric                 | Average Score |
| ---------------------- | ------------- |
| Conciseness            | 4.0           |
| Feasibility            | 4.5           |
| Effectiveness          | 4.75          |
| Helpfulness (Response) | 2.0           |

## Challenges & Next Steps

We ran into MANY challenges with this project:

1. Inability to write flexible, functional python code to wrangle data from one format to another

2. Tendency to hallucinate variables and data

3. Difficulty storing & retrieving intermediary function outputs for use, especially during final response generation

Moving forward, we will:

1. Improve the 'Write Python Code' tool to generate more functional code

2. Generate more tools such as Save to CSV, Markdown, etc. to save function output (instead of having to worry about displaying it directly)

3. Support context retention between messages

4. Enable the AI agent to clarify ambiguities (human-in-the-loop)
