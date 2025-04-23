import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agent.runner import run_agent_pipeline
from judge import eval_plan, eval_response


sample_questions = ['What are the top 3 products, per category?', 
                    "What customers should we target for our marketing efforts?",
                    "What sellers cause late delivery the most?",
                    "Is there a significant correlation between late delivery and customer reviews?"
                    ]

average_conciceness = 0
average_effectiveness = 0
average_feasibility = 0
average_helpfulness = 0
for question in sample_questions:
    plan, response = run_agent_pipeline(question, use_ui=False)
    plan_evals = eval_plan(question, plan)
    response_evals = eval_response(question, response)
    average_conciceness += plan_evals.conciceness
    average_feasibility += plan_evals.feasibility
    average_effectiveness += plan_evals.effectiveness
    average_helpfulness += response_evals.helpfulness

print("Average Conciceness Score: ", average_conciceness / len(sample_questions))
print("Average Feasibility Score: ", average_feasibility / len(sample_questions))   
print("Average Effectiveness Score: ", average_effectiveness / len(sample_questions))
print("Average Helpfulness Score: ", average_helpfulness / len(sample_questions))   


    
