"""
/assistant endpoint — LLM tool-calling "Loan Officer Assistant".

The agent has access to:
  - get_shap_explanation(applicant_id)
  - get_similar_approved_cases(applicant_id, top_k)
  - query_fairness_db(group, metric)
  - run_counterfactual(applicant_id, feature, new_value)
"""

import json
from fastapi import APIRouter
from openai import AsyncOpenAI
from app.models import ChatRequest

router = APIRouter()
client = AsyncOpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_shap_explanation",
            "description": "Retrieve full SHAP attribution breakdown for a specific applicant or the last scored applicant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string", "description": "Applicant ID or 'last'"}
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_similar_approved_cases",
            "description": "Find the K most similar applicants who were approved. Useful for explaining what changes might flip a decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["applicant_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_fairness_db",
            "description": "Retrieve approval and default rates for a demographic segment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {"type": "string", "description": "e.g. 'informal_trader', 'female', 'Kano'"},
                    "metric": {"type": "string", "enum": ["approval_rate", "default_rate", "demographic_parity", "equalized_odds"]},
                },
                "required": ["group", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_counterfactual",
            "description": "Compute how changing a specific feature value would change the risk score and decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "applicant_id": {"type": "string"},
                    "feature_name": {"type": "string"},
                    "new_value": {"type": "number"},
                },
                "required": ["applicant_id", "feature_name", "new_value"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are the CreditIQ Loan Officer Assistant for a Nigerian digital lender.
You have tool access to the credit scoring pipeline: SHAP explanations, similar-case retrieval,
fairness metrics, and counterfactual analysis.

Your responses must:
1. Be clear, factual, and jargon-free for loan officers who are not data scientists
2. Always cite the specific SHAP features and reason codes driving a decision
3. Suggest actionable paths to approval when a borrower is declined
4. Flag fairness concerns proactively
5. Reference Nigerian regulatory requirements (CBN Consumer Protection Regs 2022, NDPC Act 2023)
   when relevant to the decision being discussed

Format numbers in Nigerian context: use ₦ for amounts, reference employment types familiar
in the Nigerian market (ajo/esusu for savings groups, Opay/PalmPay for mobile money, etc.).
"""


async def call_tool(name: str, args: dict) -> str:
    """Mock tool executor — replace with real DB/model calls in production."""
    if name == "get_shap_explanation":
        return json.dumps({
            "applicant_id": args["applicant_id"],
            "risk_score": 67.3,
            "decision": "DECLINE",
            "top_shap_features": [
                {"feature": "post_loan_dti", "value": "+0.18", "label": "Post-Loan DTI 58% (threshold: 45%)"},
                {"feature": "loan_to_income_ratio", "value": "+0.14", "label": "Loan-to-Income 4.2× (ceiling: 3×)"},
                {"feature": "bill_payment_regularity", "value": "+0.09", "label": "Bill regularity 65% (poor)"},
                {"feature": "mobile_money_velocity", "value": "-0.06", "label": "Mobile money 11 txns/wk (positive)"},
            ],
            "reason_codes": ["R1: Excessive DTI", "R2: High loan-to-income", "R3: Irregular bill payments"],
        })

    elif name == "get_similar_approved_cases":
        return json.dumps({
            "similar_approved": [
                {"id": "NG-2024-8821", "employment": "gig_worker", "income": 102000, "loan": 250000, "dti": 0.38,
                 "bill_reg": 79, "mm_velocity": 14, "score": 31.2, "decision": "APPROVE"},
                {"id": "NG-2024-6234", "employment": "gig_worker", "income": 89000, "loan": 200000, "dti": 0.35,
                 "bill_reg": 82, "mm_velocity": 16, "score": 28.4, "decision": "APPROVE"},
            ],
            "key_differences": "Approved cases had 40% lower loan amounts and DTI under 40%",
        })

    elif name == "query_fairness_db":
        group_data = {
            "informal_trader": {"approval_rate": 0.22, "default_rate": 0.38, "n": 1876},
            "female": {"approval_rate": 0.47, "default_rate": 0.15, "n": 5786},
            "Kano": {"approval_rate": 0.33, "default_rate": 0.27, "n": 1876},
        }
        return json.dumps(group_data.get(args["group"], {"error": "Group not found"}))

    elif name == "run_counterfactual":
        return json.dumps({
            "original_score": 67.3,
            "original_decision": "DECLINE",
            "new_score": 54.1,
            "new_decision": "REVIEW",
            "delta": -13.2,
            "feature_changed": args["feature_name"],
            "new_value": args["new_value"],
            "note": "Changing this feature alone is not sufficient for approval but moves the case to manual review.",
        })

    return json.dumps({"error": f"Unknown tool: {name}"})


@router.post("/assistant/chat", summary="Loan Officer AI assistant with tool-calling")
async def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if request.applicant_context:
        ctx = request.applicant_context.model_dump()
        messages.append({
            "role": "system",
            "content": f"Active applicant context: {json.dumps(ctx)}",
        })

    for m in request.messages:
        messages.append({"role": m.role, "content": m.content})

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.2,
    )

    msg = response.choices[0].message

    # Agentic tool loop
    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            result = await call_tool(tc.function.name, json.loads(tc.function.arguments))
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

    return {"reply": msg.content, "tool_calls_made": []}
