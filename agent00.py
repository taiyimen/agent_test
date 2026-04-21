from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import base64
import io
import shap
from langchain_classic.memory import ConversationBufferMemory
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List, Union, TypedDict
import operator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from volcenginesdkarkruntime import Ark
import json
import time


os.environ["DEEPSEEK_API_KEY"] = ""

def invoke_with_retry(runnable, payload, max_attempts=4, base_delay=1.5):
    """
    Retry LLM/tool-call invocations on transient API connection failures.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return runnable.invoke(payload)
        except Exception as e:
            is_connection_error = (
                e.__class__.__name__ == "APIConnectionError"
                or "Connection error" in str(e)
            )
            if not is_connection_error or attempt == max_attempts:
                raise
            wait_s = base_delay * (2 ** (attempt - 1))
            print(
                f"[WARN] LLM connection failed (attempt {attempt}/{max_attempts}), "
                f"retrying in {wait_s:.1f}s..."
            )
            time.sleep(wait_s)
@tool
def aki_risk_predictor(patient_features: dict) -> dict:
    """
    预测术后急性肾损伤（AKI）的发生风险概率。
    输入为结构化的患者临床特征，输出为模型预测的 AKI 风险
    patient_feature的顺序一定要是['intraoperative Partial nephrectomy', 'Operation time',
    'OR-out_BUN','urine output':105,'Preoperative WBC', 
    'OR-out_Lac','intraoperative min_Ca++', 'liquid balance',  
    'min_MAP', 'loss blood', 'intraoperative max_K+', 'latest Scr before surgery'],如果不一致请修改
    调用后询问时都需要分析各特征对结局的贡献
    """
    aki_model=joblib.load(r"D:\agent3333\agent\model\true_model.pkl")
    scaler = joblib.load(r"D:\agent3333\agent\model\model.pklscaler.pkl")
    X = scaler.transform(pd.DataFrame([patient_features]))
    X = pd.DataFrame(X,columns=pd.DataFrame([patient_features]).columns)
    prob = aki_model.predict_proba(X)[0, 1]
    return {
        "aki_risk": float(prob)
    }

@tool
def plot_sine_wave(patient_features: dict) -> dict:
    """用于可视化病人的AKI预测结果，调用时只输出结果，不要有任何文本描述"""

    aki_model=joblib.load(r"D:\agent3333\agent\model\model.pkl")
    explainer = shap.TreeExplainer(aki_model)
    scaler = joblib.load(r"D:\agent3333\agent\model\model.pklscaler.pkl")
    X = scaler.transform(pd.DataFrame([patient_features]))
    X = pd.DataFrame(X,columns=pd.DataFrame([patient_features]).columns)
    shap_values = explainer.shap_values(X)
    shap_importance = np.abs(shap_values).mean(axis=0)
    top20_idx = np.argsort(shap_importance)[::-1][:12]
    df_list=[]

    for j in top20_idx:
        df_list.append({
                "patient": f"Patient",
                "feature": X.columns[j],
                "value": X.iloc[:,j],
                "shap":  shap_values[0][j]
            })
    df_long = pd.DataFrame(df_list)

    patients = ['Patient']
    features = df_long["feature"].unique()
    heatmat = df_long.pivot(index="feature", columns="patient", values="value")
    heatmat = heatmat[patients]
    heatmat = heatmat.reindex(index=features)
    heatmat = heatmat.astype(float)
    
    plt.figure(figsize=(15, 15))
    ax = sns.heatmap(
        heatmat, cmap="RdBu_r",
        center=0,
        linewidths=0.3,
        linecolor="black",
        cbar_kws={"label": "Standardized marker value (z-score)"}
    )
    for i, feat in enumerate(features):
        for k, p in enumerate(patients):
            row = df_long[(df_long.feature==feat) & (df_long.patient==p)].iloc[0]
            if row.shap>1.3:
                ax.scatter(
                k +0.5+row.shap/4,
                i + 0.5,a
                s = 60,
                c='black',
                cmap="coolwarm",
                vmin=-np.max(np.abs(df_long.shap)),
                vmax= np.max(np.abs(df_long.shap)),
                edgecolors="black",
            )
            elif row.shap>0.4:
                ax.scatter(
                k +0.5+row.shap/2,
                i + 0.5,
                s = 60,
                c='black',
                cmap="coolwarm",
                vmin=-np.max(np.abs(df_long.shap)),
                vmax= np.max(np.abs(df_long.shap)),
                edgecolors="black",
            )
            else:
                ax.scatter(
                    k + 0.5+row.shap,
                    i + 0.5,
                    s = 60,
                    
                    c='black',
                    cmap="coolwarm",
                    vmin=-np.max(np.abs(df_long.shap)),
                    vmax= np.max(np.abs(df_long.shap)),
                    edgecolors="black",

                )
            ax.plot(
                [k + 0.5, k + 0.5],  
                [i + 0.2, i + 0.8],   
                color="black",
                linewidth=3
            )
            
    ax.set_xticks(np.arange(len(patients)) + 0.5)
    ax.set_xticklabels(['Patient'], rotation=45, ha='right', fontsize=14)
    ax.set_yticks(np.arange(len(features)) + 0.5)
    ax.set_yticklabels(features, fontsize=14)
    plt.title("Patient feature landscape with SHAP contributions", fontsize=14)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()

    buf.seek(0)

    image_base64 = base64.b64encode(buf.read()).decode()

    return f"data:image/png;base64,{image_base64}"

@tool
def aki_shap_value(patient_features: dict) -> dict:
    """
    如果医生要求计算每个特征的贡献，使用该工具不需要提前计算风险，输入为结构化的患者临床特征，输出为模型预测的每个特征的shap值。
    调用完后询问是否需要可视化
    """

    aki_model = joblib.load(r"D:\agent3333\agent\model\model.pkl")
    scaler = joblib.load(r"D:\agent3333\agent\model\model.pklscaler.pkl")

    X = scaler.transform(pd.DataFrame([patient_features]))
    X = pd.DataFrame(X, columns=pd.DataFrame([patient_features]).columns)

    explainer = shap.TreeExplainer(aki_model)
    shap_values = explainer.shap_values(X)

    try:
        if isinstance(shap_values, list):

            arr = np.array(shap_values[-1])[0]
        else:
            arr = np.array(shap_values)[0]
    except Exception:

        return {"error": "无法解析 SHAP 输出", "raw": str(shap_values)}

    features = list(X.columns)
    shap_map = {feat: float(val) for feat, val in zip(features, arr)}
    sorted_by_abs = sorted(shap_map.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "type": "shap_table",
        "shap_values": shap_map,
        "shap_sorted": [{"feature": f, "shap": v} for f, v in sorted_by_abs]
    }

@tool
def aki_Is_serious(patient_features: dict) -> dict:
    """
    如果医生不问严重程度，不要使用该工具，仅在患者aki风险大于0.5时调用，输入为结构化的患者临床特征，输出患者患中重度aki的概率。
    """
    aki_model=joblib.load(r"D:\agent3333\agent\model\true_model_2.pkl")
    
    scaler = joblib.load(r"D:\agent3333\agent\model\model.pklscaler.pkl")
    X = scaler.transform(pd.DataFrame([patient_features]))
    X = pd.DataFrame(X,columns=pd.DataFrame([patient_features]).columns)
    prob = aki_model.predict_proba(X)[0, 1]
    return {
        "serious_risk": float(prob)
    }

import joblib
import pandas as pd

@tool
def operation_time_causal_effect(patient_features: dict, new_operation_time: float) -> dict:
    """
    分析 Operation time 改变对 AKI 风险的因果影响
    """
    bundle = joblib.load(
        r"D:\agent3333\agent\model\Operation time.pkl"
    )
    model = bundle["model"]
    # 转 dataframe
    df = pd.DataFrame([patient_features])
    confounders = bundle["confounders"]
    X = df[confounders]

    # 个体治疗效应
    identified_estimand = model.identify_effect()

    # 估计因果量
    estimate = model.estimate_effect(identified_estimand, method_name=bundle["method"])

    # 估计值就是 ATE
    ate = estimate.value

    current_operation_time = patient_features["Operation time"]
    delta = new_operation_time - current_operation_time

    predicted_change = ate * delta  # 线性近似

    return {
        "current_operation_time": current_operation_time,
        "target_operation_time": new_operation_time,
        "estimated_AKI_risk_change": float(predicted_change)
    }


@tool
def liquid_balance_causal_effect(patient_features: dict, new_liquid_balance: float) -> dict:
    """
    分析 Liquid balance 改变对 AKI 风险的因果影响
    """
    bundle = joblib.load(
        r"D:\agent3333\agent\model\liquid balance.pkl"
    )
    model = bundle["model"]
    # 转 dataframe
    df = pd.DataFrame([patient_features])
    confounders = bundle["confounders"]
    X = df[confounders]

    # 个体治疗效应
    identified_estimand = model.identify_effect()

    # 估计因果量
    estimate = model.estimate_effect(identified_estimand, method_name=bundle["method"])

    # 估计值就是 ATE
    ate = estimate.value

    current_liquid_balance = patient_features["Liquid balance"]
    delta = new_liquid_balance - current_liquid_balance

    predicted_change = ate * delta  # 线性近似

    return {
        "current_liquid_balance": current_liquid_balance,
        "target_liquid_balance": new_liquid_balance,
        "estimated_AKI_risk_change": float(predicted_change)
    }

@tool
def intraoperative_Partial_nephrectomy_causal_effect(patient_features: dict, new_intraoperative_Partial_nephrectomy: float) -> dict:
    """
    分析 Intraoperative Partial Nephrectomy 改变对 AKI 风险的因果影响
    """
    bundle = joblib.load(
        r"D:\agent3333\agent\model\intraoperative Partial nephrectomy.pkl"
    )
    model = bundle["model"]
    # 转 dataframe
    df = pd.DataFrame([patient_features])
    confounders = bundle["confounders"]
    X = df[confounders]

    # 个体治疗效应
    identified_estimand = model.identify_effect()

    # 估计因果量
    estimate = model.estimate_effect(identified_estimand, method_name=bundle["method"])

    ate = estimate.value

    current_intraoperative_Partial_nephrectomy = patient_features["intraoperative Partial nephrectomy"]
    delta = new_intraoperative_Partial_nephrectomy - current_intraoperative_Partial_nephrectomy

    predicted_change = ate * delta  

    return {
        "current_intraoperative_Partial_nephrectomy": current_intraoperative_Partial_nephrectomy,
        "target_intraoperative_Partial_nephrectomy": new_intraoperative_Partial_nephrectomy,
        "estimated_AKI_risk_change": float(predicted_change)
    }


tools = [
    aki_risk_predictor, 
    aki_shap_value, 
    aki_Is_serious, 
    plot_sine_wave,
    operation_time_causal_effect, 
    liquid_balance_causal_effect, 
    intraoperative_Partial_nephrectomy_causal_effect
]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    file: None
    current_risk: float = None
    patient_data: dict = None
#新增统一特征提取函数

def extract_features_with_doubao(content: str, file_type: str = None, image_base64: str = None) -> dict:
    """
    使用豆包模型从文本/图片中提取指定特征
    """
    feature_map = {
        "intraoperative Partial nephrectomy": "肾部分切除术",
        "Operation time": "手术时间",
        "OR-out_BUN": "OR-out_BUN",
        "urine output": "尿量",
        "Preoperative WBC": "术前白细胞计数",
        "OR-out_Lac": "OR-out_Lac",
        "intraoperative min_Ca++": "术中最低钙离子浓度",
        "liquid balance": "液体平衡",
        "min_MAP": "最低平均动脉压",
        "loss blood": "失血量",
        "intraoperative max_K+": "术中最高钾离子浓度",
        "latest Scr before surgery": "手术前最新血清肌酐"
    }
    prompt = f"""
请从以下输入中提取患者临床特征，输出为 JSON 格式，键为特征英文名，值为数值。
特征列表：
{json.dumps(feature_map, ensure_ascii=False, indent=2)}

要求：
- 如果某个特征无法提取，该键不出现
- 数值类型统一为数字（整数或浮点数）
- 只输出 JSON，不要其他文字

输入内容：
{content}
"""
    try:
        if file_type == "image" and image_base64:
            response = client.responses.create(
                model="doubao-seed-2-0-pro-260215",
                input=[
                    {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                    {"role": "user", "content": [
                        {"type": "input_image", "image_url": f"data:image/png;base64,{image_base64}"},
                        {"type": "input_text", "text": content}
                    ]}
                ]
            )
            output_text = ""
            for item in response.output:
                if item.type == "message":
                    for c in item.content:
                        if c.type == "output_text":
                            output_text += c.text
        else:
            response = client.chat.completions.create(
                model="doubao-seed-2-0-pro-260215",
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ]
            )
            output_text = response.choices[0].message.content

        # 解析 JSON
        json_str = output_text.strip()
        if json_str.startswith("```json"):
            json_str = json_str.split("```json")[1].split("```")[0]
        features = json.loads(json_str)
        # 转换为数值
        for k, v in features.items():
            try:
                features[k] = float(v)
            except:
                pass
        return features
    except Exception as e:
        print(f"[WARN] Feature extraction failed: {e}")
        return {}



tool_node = ToolNode(tools)


#数据处理智能体：根据分析智能体的判断，进行数据的提取和处理，如果有图片，就进行图片信息的提取，如果是表格，就进行表格信息的提取，最终输出结构化的数据，格式为{'intraoperative Partial nephrectomy': 0,'loss blood':100,...}，如果没有数据，就直接输出空
def Data_processing_agent(state: AgentState):
    messages = state['messages']
    file = state.get('file')
    patient_data = state.get('patient_data', {})
    
    # 获取用户最新输入
    user_input = ""
    for msg in reversed(messages):
        if msg.type == "human":
            user_input = msg.content
            break
    
    extracted_features = {}
    
    if file is not None:
        # 处理文件
        if file.get('mime', '').startswith("image/"):
            # 图片
            image_base64 = file.get('data')
            extracted_features = extract_features_with_doubao(
                user_input, file_type="image", image_base64=image_base64
            )
        elif file.get('mime') in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ]:
            # Excel 文件：从 base64 解码读取
            import io
            file_data = file.get('data')
            decoded = base64.b64decode(file_data)  # bytes
            df = pd.read_excel(io.BytesIO(decoded))
            # 将表格转为 JSON 字符串
            table_text = df.to_json(orient="records", force_ascii=False)
            # 构造完整输入
            content = f"表格数据如下：\n{table_text}\n用户问题：{user_input}"
            extracted_features = extract_features_with_doubao(content, file_type="text")
        else:
            # 其他文件类型暂不支持
            extracted_features = {}
    else:
        # 纯文本输入
        extracted_features = extract_features_with_doubao(user_input, file_type="text")
    
    # 合并特征
    patient_data.update(extracted_features)
    
    # 反馈消息
    if extracted_features:
        feedback = f"已从输入中提取患者特征：{json.dumps(extracted_features, ensure_ascii=False)}"
    else:
        feedback = "未能从输入中提取有效特征，请确保提供的数据包含所需信息。"
    
    new_msg = AIMessage(content=feedback)
    return {
        "messages": [new_msg],
        "patient_data": patient_data
    }



def analyst_agent(state: AgentState):#分析智能体，根据自然语言信息，分析出来要执行的任务和要调用的工具
    messages = state['messages']

    PROMPT = """
    你是一个任务分析智能体，你的任务是从使用者提出的问题中，分析出想要实现的任务(可能有AKI概率计算，特征贡献度的计算，VTE，因果分析，AKI严重程度计算等等，如果使用者只是简单打招呼对话聊天，task为other),判断是否有数据,如果有，is_data为1，并将数据提取到'data'
    请按以下 JSON 格式输出：
    {
    "is_data": 0 or 1,
    "task": AKI or VTE or other,
    'data':{'intraoperative Partial nephrectomy':0,'loss blood':100,...}
    }
    """
    system_message = [SystemMessage(content=PROMPT)]
    full_messages=system_message+messages
    try:
        response = invoke_with_retry(llm, full_messages)
    except Exception as e:
        print(f"[ERROR] analyst_agent failed to call LLM: {e}")
        fallback = {
            "is_data": 0,
            "task": "other",
            "data": {},
            "error": "llm_connection_error"
        }
        response = AIMessage(content=json.dumps(fallback, ensure_ascii=False))
    return {"messages": [response]}


def route_from_analyst(state: AgentState):
    last_message = state['messages'][-1]
    file = state['file']
    try:
        result = json.loads(last_message.content)
    except Exception:
        result = {"is_data": 0, "task": "other", "data": {}}
    if file is not None or (result and result.get("is_data") == 1):
        return 'Data_processing'
    return 'Output'

#工具调用智能体，根据分析智能体的判断，调用相应的工具进行计算和可视化，输入为结构化的数据，输出为计算结果或者可视化结果，如果没有数据，就直接输出文本结果
def Calculation_agent(state: AgentState):
    messages = state['messages']
    patient_data = state.get('patient_data', {})
    
    # 改进点: 将提取的特征以系统消息形式注入上下文，帮助 LLM 调用工具
    context = ""
    if patient_data:
        context = f"当前患者特征（已提取）：{json.dumps(patient_data, ensure_ascii=False)}\n"
        # 如果已有特征，优先使用这些特征作为工具输入
        system_msg = SystemMessage(content=context)
        messages_with_context = [system_msg] + messages
    else:
        messages_with_context = messages
    
    tool_call = llm2.bind_tools(tools)
    try:
        responses = invoke_with_retry(tool_call, messages_with_context)
    except Exception as e:
        print(f"[ERROR] Calculation_agent failed: {e}")
        responses = AIMessage(content="模型连接异常，请稍后再试。")
    return {"messages": [responses]}


def route_from_calculation(state: AgentState):
    last_message = state['messages'][-1]        
    #f last_message.tool_calls:
    #   return "tools"
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "Output"

import json
from langchain_core.messages import SystemMessage

def Output_agent(state: AgentState):
    """
    信息整合智能体：
    - 基于 analyst 的 task 判断任务类型
    - 医疗任务 → 转换为患者可理解语言
    - 普通聊天 → 正常回答（基于事实）
    """

    messages = state["messages"]
    



    # ----------------------
    # 1️⃣ 获取用户问题
    # ----------------------
    user_query = ""
    for msg in messages:
        if msg.type == "human":
            user_query = msg.content
            break

    # ----------------------
    # 2️⃣ 获取 task（来自 analyst_agent）
    # ----------------------
    for msg in reversed(messages):
        if msg.type == "ai":
            try:
                data = json.loads(msg.content)
                if isinstance(data, dict) and "task" in data:
                    task = data["task"]
                    break
            except:
                continue

    # ----------------------
    # 3️⃣ 获取计算结果
    # ----------------------
    last_message = messages[-1]
    result = last_message.content

    # ----------------------
    # 4️⃣ 构造 Prompt
    # ----------------------
    PROMPT = f"""
你是一个信息整合智能体。
你的任务是将系统输出转换为用户能够理解的内容。

----------------------
【用户问题】
{user_query}

【系统输出】
{result}

【任务类型】
{task}
----------------------

请严格按照以下规则执行：

======================

✅ 情况1：如果 task = other（普通聊天）

- 正常回答用户问题
- 必须基于事实
- 不允许编造信息
- 表达自然、清晰

======================

✅ 情况2：如果 task ≠ other（医学相关）

请按以下格式输出：

任务：

1. 如果是AKI风险预测，输出预测的AKI风险概率（0-1之间的小数）
如果是特征对AKI的贡献度分析，输出每个特征的shap值（具体的特征名称+ shap值）
如果是AKI严重程度分析，输出中重度AKI的风险概率
如果是因果分析，输出改变某个特征后AKI风险的变化量
3. 用简单易懂的语言解释含义（避免或解释专业术语）
4. 给出合理建议（如就医、观察、复查等）
5. 语气客观温和，避免引发恐慌

请按以下结构输出：


【结论】
（根据患者输入信息和模型计算结果，给出结论）

【通俗解释】
（用生活化语言解释）

【建议】
（2-4条可执行建议）

【重要说明】
（结果仅供参考，不能替代医生诊断）

注意：

避免专业术语或需解释
不做确定性诊断
保证普通人易理解

"""

    # ----------------------
    # 5️⃣ 调用模型
    # ----------------------
    try:
        response = invoke_with_retry(llm3, [SystemMessage(content=PROMPT)])
    except Exception as e:
        print(f"[ERROR] Output_agent failed to call LLM: {e}")
        response = AIMessage(content="Model connection is unstable right now. Please try again shortly.")

    return {
        "messages": [response]
    }

def del_node(state: AgentState):
    state["messages"].pop(-3)
   
    return {}


workflow = StateGraph(AgentState)

workflow.add_node("Data_processing", Data_processing_agent)
workflow.add_node("analyst", analyst_agent)
workflow.add_node("Calculation", Calculation_agent)
workflow.add_node("Output", Output_agent)#输出智能体
workflow.add_node("tools", tool_node)
workflow.add_node("del_node", del_node)#删除工具调用信息的中间节点

workflow.set_entry_point("analyst")

workflow.add_conditional_edges(
    "analyst",
    route_from_analyst,
    {
        "Data_processing": "Data_processing",
        "Output": "Output"
    }
)
workflow.add_edge("Calculation", "Output")#如果分析智能体没有分析出任务，就直接输出文本结果
workflow.add_edge("Data_processing", "Calculation")
workflow.add_conditional_edges(
    'Calculation',
    route_from_calculation,
    {   
        "tools": "tools",
        "Output": "Output"
 
    }
)



workflow.add_edge("tools", "del_node")#工具调用完后直接输出结果
workflow.add_edge("del_node", "Output")#输出智能体输出结果后结束
llm = ChatDeepSeek(
     model="deepseek-chat",  
     temperature=0.5
 )
llm2 = ChatDeepSeek(
     model="deepseek-chat",  
     temperature=0.5
 )
llm3 = ChatDeepSeek(
     model="deepseek-chat",  
     temperature=0.5
 )

client = Ark(base_url='https://ark.cn-beijing.volces.com/api/v3',api_key='f57a148e-b69e-45f8-8cca-0091d444a3e2')

app_graph = workflow.compile()


# SYSTEM_PROMPT = """
# 你是一名临床决策支持智能体（CDS Agent）。
# 你不做诊断，不给出治疗方案。
# 回答要简短，禁止回答除医生问以外的问题
# 回答完问题，询问是否需要可视化
# """
# memory = ConversationBufferMemory(
#     memory_key="chat_history",
#     return_messages=True
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", SYSTEM_PROMPT),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}")
#     ]
# )
# llm = ChatDeepSeek(
#     model="deepseek-chat",  
#     temperature=0.5
# )

# agent = create_tool_calling_agent(
#     llm=llm,
#     tools=[plot_sine_wave,aki_risk_predictor,aki_shap_value,aki_Is_serious,
#     operation_time_causal_effect,liquid_balance_causal_effect,intraoperative_Partial_nephrectomy_causal_effect],
#     prompt=prompt
# )
# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=[plot_sine_wave,aki_risk_predictor,aki_shap_value,aki_Is_serious,
#     operation_time_causal_effect,liquid_balance_causal_effect,intraoperative_Partial_nephrectomy_causal_effect],
#     memory=memory,
#     verbose=True,
#     return_intermediate_steps=True
# )

# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# class ChatRequest(BaseModel):
#     message: str


# @app.post("/chat")
# def chat(req: ChatRequest):

#     result = agent_executor.invoke(
#         {"input": req.message}
#     )
#     output = result["output"]
#     image_data=None

#     if "intermediate_steps" in result:
#         for action, observation in result["intermediate_steps"]:

#             if isinstance(observation, str) and "data:image" in observation:
#                 image_data = observation
#                 break 

#             elif isinstance(observation, dict) and "image" in observation:
#                 image_data = observation["image"]
#                 break

#     if image_data:
#         return {
#             "type": "image",
#             "response": image_data,
#             "text": output 
#         }

#     return {
#         "type":"text",
#         "response":output
#     }


import base64

def load_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")




def handle_chat(message: str, file: dict = None) -> dict:
    """
    对外调用接口：把用户消息（可选带 file）传入 agent graph，返回 dict {type: 'text'|'image', response: str}
    """
    # 构造临时状态（复制持久状态以保持会话）
    state = {
        "messages": [HumanMessage(content=message)] + [m for m in persistent_state.get("messages", [])],
        "file": file or persistent_state.get("file"),
        "patient_data": persistent_state.get("patient_data", {}),
        "current_risk": persistent_state.get("current_risk")
    }

    try:
        final_state = app_graph.invoke(state)
    except Exception as e:
        return {"type": "text", "response": f"Agent 执行错误: {str(e)}"}

    # 更新会话中的 messages（去掉 ToolMessage）
    messages = [m for m in final_state.get('messages', []) if not isinstance(m, ToolMessage)]
    persistent_state["messages"] = messages

    output_text = ""
    image_data = None

    for msg in reversed(final_state.get("messages", [])):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            output_text = msg.content
            break

    for msg in final_state.get("messages", []):
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if "data:image" in msg.content:
                image_data = msg.content
                break

    if image_data:
        return {"type": "image", "response": image_data, "text": output_text}

    return {"type": "text", "response": output_text}

persistent_state = {"messages": [],
    'file':None,
    "current_risk": None,
    "patient_data": {}
    }

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from fastapi import UploadFile, File,Form
@app.post("/chat")
async def chat(file: UploadFile = File(None),   # 文件（可选）
    message: str = Form(None)  ):

    if 'image' in file.headers['content-type']:
        content = await file.read()
        file = base64.b64encode(content).decode("utf-8")
        persistent_state["file"] = {
                "type": "image",
                "data": file,
                "mime": "image/png"
            }
    persistent_state["messages"].append(HumanMessage(content=message))

    final_state = app_graph.invoke(persistent_state)
    
    output_text = ""
    image_data = None
    
    # 遍历消息寻找最后的结果和可能生成的图片
    for msg in reversed(final_state["messages"]):
        if msg.type == "ai" and msg.content:
            output_text = msg.content
            break

    # 在所有工具输出中寻找图片数据
    for msg in final_state["messages"]:
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            if "data:image/png;base64," in msg.content:
                image_data = msg.content
                break

    if image_data:
        return {
            "type": "image",
            "response": image_data,
            "text": output_text
        }

    return {
        "type": "text",
        "response": output_text
    }
uvicorn.run(app, host="0.0.0.0", port=8000)


# if __name__ == "__main__":
#     print("AKI 智能体已启动（输入 exit 退出）")

#     while True:
#         u_input = input("\n医生：")

#         if u_input.startswith("img:"):
#             img_path = u_input.replace("img:", "").strip()

#             image_base64 = load_image_as_base64(img_path)

#             persistent_state["file"] = {
#                 "type": "image",
#                 "data": image_base64,
#                 "mime": "image/png"
#             }

#             print("✅ 已加载图片")

#             continue
#         if u_input.startswith("xlsx:"):
#             xlsx_path = u_input.replace("xlsx:", "").strip()
#             with open(xlsx_path, "rb") as f:
#                 file_data = base64.b64encode(f.read()).decode("utf-8")
#             persistent_state["file"] = {
#                 "type": "file",
#                 "data": file_data,
#                 "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#             }
#             print("✅ 已加载Excel文件")
#             continue


#         if u_input.lower() in ["exit", "quit"]:
#             break

#         persistent_state["messages"].append(HumanMessage(content=u_input))

#         final_state = app_graph.invoke(persistent_state)

#         messages = [m for m in final_state['messages'] if not isinstance(m, ToolMessage)]
#         persistent_state["messages"] = messages
#         output_text = ""
#         image_data = None

#         for msg in reversed(final_state["messages"]):
#             if msg.type == "ai" and msg.content:
#                 output_text = msg.content
#                 break

#         for msg in final_state["messages"]:
#             if hasattr(msg, 'content') and isinstance(msg.content, str):
#                 if "data:image" in msg.content:
#                     image_data = msg.content
#                     break

#         if image_data:
#             print(image_data)

#         print(output_text)

#         print("\nAKI 智能体：")


