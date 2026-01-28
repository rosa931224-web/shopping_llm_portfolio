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
            
# 4. 사용자 질문 입력
if prompt := st.chat_input("질문을 입력하세요 (예: 속성별 후기 분포를 차트로 그려줘)"):
    # 사용자 메시지 화면 출력 및 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("지윤님의 데이터를 분석 중입니다..."):
            try:
                # GPT에게 한국어 답변 요청
                full_prompt = f"{prompt} (모든 답변은 친절한 한국어로 해주세요)"
                response = agent_executor.invoke({"input": full_prompt})
                answer = response["output"]
                st.markdown(answer)
                
                # 이번 답변을 위한 딕셔너리 생성
                new_msg = {"role": "assistant", "content": answer}
                
                # [지능형 필터링] GPT가 판단한 데이터를 기반으로 차트 생성
                if any(keyword in prompt for keyword in ["그래프", "차트", "그려줘", "시각화"]):
                    # 1. GPT에게 현재 질문에서 어떤 데이터를 제외해야 하는지 물어봅니다.
                    filter_query = f"현재 데이터에서 사용자가 제외하고 싶어 하는 속성이 뭐야? 단어만 말해줘. (없으면 '없음')"
                    exclude_target = agent_executor.invoke({"input": f"'{prompt}' 이 질문에서 제외할 항목이 뭐야?"})["output"]
                    
                    table_name = db.get_usable_table_names()[0]
                    df = pd.read_sql(f"SELECT * FROM {table_name}", db._engine)
                    
                    # 2. GPT가 말한 단어가 포함된 행을 알아서 제거 (동적 필터링)
                    display_df = df.copy()
                    if "없음" not in exclude_target:
                        # 여러 개를 뺄 수도 있으니 유연하게 처리
                        targets = [t.strip() for t in exclude_target.replace("'", "").split(",")]
                        display_df = display_df[~display_df['속성'].isin(targets)]
                        chart_title_suffix = f" ({', '.join(targets)} 제외)"
                    else:
                        chart_title_suffix = ""

                    # 3. 색상 및 차트 생성 (기존 로직 유지하되 GPT가 거른 데이터 사용)
                    sentiment_colors = {'긍정': '#636EFA', '부정': '#EF553B', '중립': '#ABAFB3'}
                    
                    if any(k in prompt for k in ["긍부정", "만족도"]):
                        fig = px.histogram(display_df, x='속성', color='감성', barmode='group',
                                         title=f"📊 속성별 긍부정 분포{chart_title_suffix}",
                                         color_discrete_map=sentiment_colors)
                    else:
                        chart_df = display_df['속성'].value_counts().reset_index()
                        fig = px.bar(chart_df, x='속성', y='count', color='속성',
                                   color_discrete_sequence=px.colors.qualitative.Pastel,
                                   title=f"🌈 속성별 분포{chart_title_suffix}")

                    fig.update_layout(template="plotly_white")
                    new_msg["chart"] = fig

                # # [자동 시각화 로직]
                # if any(keyword in prompt for keyword in ["그래프", "차트", "그려줘", "시각화"]):
                #     table_names = db.get_usable_table_names()
                #     if table_names:
                #         table_name = table_names[0]
                #         df = pd.read_sql(f"SELECT * FROM {table_name}", db._engine)
                        
                #         # '기타' 제외 로직 유지
                #         display_df = df[df['속성'] != '기타'].copy() if ("기타" in prompt and ("제외" in prompt or "빼고" in prompt)) else df.copy()

                #         # --- 색상 설정 ---
                #         # 긍부정 색상 고정 (긍정: 파랑, 부정: 빨강, 중립: 회색)
                #         sentiment_colors = {'긍정': '#636EFA', '부정': '#EF553B', '중립': '#ABAFB3'}

                #         # 1. 긍부정(감성) 분석 요청 시
                #         if any(k in prompt for k in ["긍부정", "긍정", "부정", "만족도"]):
                #             fig = px.histogram(
                #                 display_df, x='속성', color='감성',
                #                 barmode='group', # 속성별로 긍부정 막대를 나란히 표시
                #                 title="📊 속성별 긍부정 분포 (지윤의 분석 비서)",
                #                 color_discrete_map=sentiment_colors, # [핵심] 긍부정 색상 고정
                #                 labels={'속성': '카테고리', 'count': '후기 수', '감성': '상태'}
                #             )
                        
                #         # 2. 일반 속성 분석 요청 시 (속성마다 색상 다르게)
                #         elif "속성" in prompt or "분포" in prompt:
                #             chart_df = display_df['속성'].value_counts().reset_index()
                #             fig = px.bar(
                #                 chart_df, x='속성', y='count', 
                #                 color='속성', # [핵심] 속성마다 다른 색상
                #                 color_discrete_sequence=px.colors.qualitative.Pastel,
                #                 title="🌈 속성별 후기 분포 (파스텔 테마)",
                #                 labels={'속성': '카테고리', 'count': '후기 수'}
                #             )
                        
                #         # 3. 기타(버블/파이 등) 요청 시 기본 로직 유지
                #         elif "버블" in prompt:
                #             bubble_df = display_df.groupby('속성').agg({'리뷰': 'count', '감성': lambda x: (x == '긍정').mean()}).reset_index().rename(columns={'리뷰': '건수', '감성': '긍정률'})
                #             fig = px.scatter(bubble_df, x="속성", y="긍정률", size="건수", color="속성", size_max=60, title="🎈 실시간 버블 분석")
                        
                #         else:
                #             fig = px.pie(display_df, names='감성', color='감성', color_discrete_map=sentiment_colors, title="🍕 전체 긍부정 비율")

                #         # 디자인 마무리
                #         fig.update_layout(template="plotly_white", showlegend=True)
                #         new_msg["chart"] = fig
                
                st.session_state.messages.append(new_msg)
                st.rerun()

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")