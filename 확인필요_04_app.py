import streamlit as st
import pandas as pd
import plotly.express as px
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# 1. 환경 설정 및 UI 초기화
load_dotenv()
st.set_page_config(page_title="지윤의 AI 분석 비서", layout="wide")

# DB 파일 경로 (본인의 파일명 확인!)
DB_PATH = "sqlite:///shopping_reviews.db"

@st.cache_resource
def init_agent():
    db = SQLDatabase.from_uri(DB_PATH)
    # GPT-4o 설정
    llm = ChatOpenAI(
        model="gpt-4o", 
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0
    )
    # SQL 에이전트 생성
    agent = create_sql_agent(
        llm, db=db, agent_type="openai-tools", verbose=True
    )
    return agent, db

agent_executor, db = init_agent()

# 2.사이드바 및 헤더 (한글화)
st.title("🤖 쇼핑 후기 AI 자동 분석 시스템")
st.markdown("---")
st.sidebar.header("📋 분석 가이드")
st.sidebar.info("""
이렇게 물어보세요!
1. "전체 후기 개수가 몇 개야?"
2. "속성별 후기 분포를 차트로 그려줘"
3. "가장 불만이 많은 속성은 뭐야?"
""")

# 3. 채팅 세션 관리
if "messages" not in st.session_state:
    st.session_state.messages = []

# [교정 완료] 저장된 메시지와 차트를 순서대로 다시 그리기
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # 차트가 포함된 메시지라면 고유한 key(이름표)를 붙여서 출력
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True, key=f"history_{idx}")
            
# 4. 사용자 질문 입력 및 분석 로직 (전체 교체 구간)
if prompt := st.chat_input("질문을 입력하세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("지윤님의 요청을 분석 중입니다..."):
            try:
                # --- [1] 맥락 분석: 제외할 단어 및 텍스트 출력 여부 판단 ---
                # 최근 3개의 대화를 가져와 GPT에게 상황을 물어봅니다.
                recent_context = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-3:]])
                
                analysis_llm = ChatOpenAI(model="gpt-4o", temperature=0)
                analysis_res = analysis_llm.invoke(
                    f"이전 대화:\n{recent_context}\n\n"
                    f"현재 질문: '{prompt}'\n"
                    "---"
                    "위 대화를 보고 다음 두 가지만 딱 정해줘.\n"
                    "1. 제외할 속성 단어 (기타/배송 등, 다시 포함하라면 '없음')\n"
                    "2. 답변 텍스트(리뷰 등) 출력 여부 (보여주지 말라면 '숨김', 아니면 '표시')\n"
                    "출력 형식: 제외단어, 출력여부"
                ).content.strip()

                # 분석 결과 분리 (예: '기타, 숨김' 또는 '없음, 표시')
                exclude_word, visibility = [x.strip() for x in analysis_res.split(",")]

                # --- [2] GPT 답변 생성 (DB 쿼리 수행) ---
                full_prompt = f"{prompt} (모든 답변은 친절한 한국어로 해주세요)"
                response = agent_executor.invoke({"input": full_prompt})
                answer = response["output"]

                # 텍스트 출력 제어
                if visibility == "표시":
                    st.markdown(answer)
                else:
                    st.info("📊 요청하신 대로 텍스트 설명은 생략하고 분석 차트만 갱신합니다.")
                
                new_msg = {"role": "assistant", "content": answer, "visibility": visibility}

                # --- [3] 지능형 시각화 (추출된 exclude_word 반영) ---
                if any(keyword in prompt for keyword in ["그래프", "차트", "그려줘", "시각화"]):
                    table_name = db.get_usable_table_names()[0]
                    df = pd.read_sql(f"SELECT * FROM {table_name}", db._engine)
                    
                    display_df = df.copy()
                    chart_title_suffix = ""
                    
                    # GPT가 판단한 제외 단어 적용
                    if "없음" not in exclude_word:
                        display_df = display_df[display_df['속성'] != exclude_word]
                        chart_title_suffix = f" ({exclude_word} 제외)"
                    
                    # 색상 설정
                    sentiment_colors = {'긍정': '#636EFA', '부정': '#EF553B', '중립': '#ABAFB3'}

                    # 차트 종류 결정
                    if any(k in prompt for k in ["긍부정", "만족도"]):
                        fig = px.histogram(display_df, x='속성', color='감성', barmode='group',
                                         title=f"📊 속성별 긍부정 분포{chart_title_suffix}",
                                         color_discrete_map=sentiment_colors)
                    elif "파이" in prompt or "원형" in prompt:
                        fig = px.pie(display_df, names='속성', title=f"🍕 속성별 비율{chart_title_suffix}", hole=0.3,
                                   color_discrete_sequence=px.colors.qualitative.Pastel)
                    else:
                        chart_df = display_df['속성'].value_counts().reset_index()
                        fig = px.bar(chart_df, x='속성', y='count', color='속성', 
                                   title=f"📊 주요 속성 분포{chart_title_suffix}",
                                   color_discrete_sequence=px.colors.qualitative.Pastel)

                    fig.update_layout(template="plotly_white")
                    new_msg["chart"] = fig
                
                # 최종 저장 및 리런
                st.session_state.messages.append(new_msg)
                st.rerun()

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")