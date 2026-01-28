import streamlit as st
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import plotly.express as px
import plotly.graph_objects as go

# 워드클라우드 및 matplotlib 선택적 import (없으면 해당 기능만 비활성화)
WORDCLOUD_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    pass

# 1. 환경 설정 및 데이터 로드
load_dotenv()
st.set_page_config(page_title="AI 자동 분석 시스템", layout="wide")

@st.cache_resource
def load_data():
    # 경로 정합성을 위해 sqlite3 직접 연결
    conn = sqlite3.connect("shopping_reviews.db")
    df = pd.read_sql("SELECT * FROM naver_reviews", conn)
    conn.close()
    return df

df = load_data()

# 사이드바 및 헤더 
st.title("AI 자동 분석 시스템")
st.markdown("---")
st.sidebar.header("📋 분석 가이드")
st.sidebar.info("""
이렇게 물어보세요!
1. "전체 후기 개수가 몇 개야?"
2. "속성별 후기 분포를 차트로 그려줘"
3. "가장 불만이 많은 속성은 뭐야?"
""")

# 2. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_chart_attrs" not in st.session_state:
    st.session_state.last_chart_attrs = None  # 이전 차트에서 사용한 속성들
if "last_chart_type" not in st.session_state:
    st.session_state.last_chart_type = None  # 이전 차트 유형 (simple/sentiment)

# 3. 화면 UI 및 이전 기록 출력
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # plotly 차트 복원 (JSON 또는 기존 Figure 객체, 리스트 지원)
        if "chart_json" in message and message["chart_json"] is not None:
            chart_data = message["chart_json"]
            # 리스트인 경우 여러 차트 표시
            if isinstance(chart_data, list):
                for j, chart_j in enumerate(chart_data):
                    fig = go.Figure(chart_j)
                    st.plotly_chart(fig, use_container_width=True, key=f"hist_{i}_{j}")
            else:
                fig = go.Figure(chart_data)
                st.plotly_chart(fig, use_container_width=True, key=f"hist_{i}")
        elif "chart" in message and message["chart"] is not None:
            # 이전 버전 호환 (Figure 객체로 저장된 경우)
            st.plotly_chart(message["chart"], use_container_width=True, key=f"old_{i}")
        # 워드클라우드 이미지 복원 (base64)
        if "wordcloud_img" in message and message["wordcloud_img"] is not None:
            import base64
            st.image(f"data:image/png;base64,{message['wordcloud_img']}")

# 4. 분석 로직 (복잡한 모듈 없이 ChatOpenAI만 사용)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

if prompt := st.chat_input("질문을 입력하세요!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # 변수 초기화 (스코프 문제 방지)
        fig = None
        analysis_res = None
        
        # 1. 상태 표시 바 (진행 상황을 시각적으로 보여줌)
        with st.status("🚀 분석을 시작합니다...", expanded=True) as status:
            try:
                # 상태 메시지를 한 줄로 표시하기 위한 placeholder
                status_text = st.empty()
                
                # [Step 1] 차트 유형 및 분석 방향 결정
                status_text.write("🔍 질문을 분석 중입니다...")
                chart_decision_msg = [
                    ("system", """사용자 질문을 분석하여 아래 형식으로만 답하세요:
차트유형: [속성분포/감성분포/속성별감성/없음]
제외단어: [제외할 속성 단어 또는 '없음']

예시:
- "속성별 감성을 차트로 그려줘" → 차트유형: 속성별감성, 제외단어: 없음
- "감성 분포 보여줘" → 차트유형: 감성분포, 제외단어: 없음
- "속성별 분포 그려줘" → 차트유형: 속성분포, 제외단어: 없음
- "전체 후기 개수가 몇개야?" → 차트유형: 없음, 제외단어: 없음"""),
                    ("user", prompt)
                ]
                decision = llm.invoke(chart_decision_msg).content.strip()
                
                # 결정 파싱 (더 유연하게)
                chart_type = "속성분포"  # 기본값
                exclude_word = "없음"
                for line in decision.split('\n'):
                    if "차트유형" in line or "차트 유형" in line:
                        raw_type = line.split(":")[-1].strip()
                        # 대괄호, 띄어쓰기 제거
                        raw_type = raw_type.replace("[", "").replace("]", "").replace(" ", "")
                        chart_type = raw_type
                    if "제외단어" in line or "제외 단어" in line:
                        exclude_word = line.split(":")[-1].strip().replace("[", "").replace("]", "")
                
                # [Step 2] 데이터 처리
                status_text.write(f"📊 데이터를 집계하고 필터링합니다...")
                plot_df = df.copy()
                if "없음" not in exclude_word:
                    plot_df = plot_df[plot_df['속성'] != exclude_word]
                
                # [Step 3] 차트 유형에 따른 시각화 생성
                status_text.write("📈 맞춤형 차트를 생성하고 있습니다...")
                
                # Top N 필터링 감지
                import re
                top_n = None
                top_match = re.search(r'top\s*(\d+)|상위\s*(\d+)|(\d+)\s*개만', prompt.lower())
                if top_match:
                    top_n = int(top_match.group(1) or top_match.group(2) or top_match.group(3))
                
                # 감성 필터링 감지
                sentiment_filter = None
                if "부정" in prompt:
                    sentiment_filter = "부정"
                elif "긍정" in prompt:
                    sentiment_filter = "긍정"
                elif "중립" in prompt:
                    sentiment_filter = "중립"
                
                # 특정 속성명 감지
                all_attributes = df['속성'].unique().tolist()
                mentioned_attrs = [attr for attr in all_attributes if attr in prompt]
                
                # "지우고", "제외하고" 패턴 감지 - 이전 차트에서 특정 속성 제외
                exclude_keywords = ["지우고", "제외하고", "빼고", "없이", "제외"]
                is_chart_modification = False  # 이전 차트 수정 여부
                
                if any(kw in prompt for kw in exclude_keywords) and st.session_state.last_chart_attrs:
                    # 이전 차트 속성에서 언급된 속성 제외
                    prev_attrs = st.session_state.last_chart_attrs.copy()
                    for attr in mentioned_attrs:
                        if attr in prev_attrs:
                            prev_attrs.remove(attr)
                    if prev_attrs:
                        plot_df = plot_df[plot_df['속성'].isin(prev_attrs)]
                        mentioned_attrs = prev_attrs
                        is_chart_modification = True  # 이전 차트 수정 모드
                elif mentioned_attrs:
                    # 특정 속성이 언급되면 해당 속성만 필터링
                    filtered_df = df[df['속성'].isin(mentioned_attrs)]
                    if len(filtered_df) > 0:
                        plot_df = filtered_df
                    else:
                        st.info(f"ℹ️ '{', '.join(mentioned_attrs)}' 속성 데이터를 찾을 수 없어 전체 데이터를 사용합니다.")
                
                # 사용자 질문에서 직접 차트 유형 결정
                wordcloud_img_base64 = None  # 워드클라우드 이미지 초기화
                
                # 이전 차트 수정 모드일 때 - 이전 차트 타입 유지
                if is_chart_modification and st.session_state.last_chart_type:
                    if st.session_state.last_chart_type == "simple":
                        # 단순 속성 분포 차트 (이전과 동일)
                        counts = plot_df['속성'].value_counts().reset_index()
                        counts.columns = ['속성', '리뷰수']
                        fig = px.bar(counts, x='속성', y='리뷰수', color='속성', 
                                     title="📊 속성 분포",
                                     text='리뷰수',
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_traces(textposition='outside')
                    elif st.session_state.last_chart_type == "pie":
                        # 파이차트 (이전과 동일)
                        counts = plot_df['속성'].value_counts().reset_index()
                        counts.columns = ['속성', '리뷰수']
                        fig = px.pie(counts, names='속성', values='리뷰수', 
                                     title="📊 속성 분포 (파이차트)",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                        fig.update_traces(textposition='inside', textinfo='label+value+percent')
                    else:  # sentiment
                        # 속성별 감성 분포 차트
                        counts = plot_df.groupby(['속성', '감성']).size().reset_index(name='리뷰수')
                        fig = px.bar(counts, x='속성', y='리뷰수', color='감성', 
                                     barmode='group',
                                     title="📊 속성별 감성 분포",
                                     text='리뷰수',
                                     color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                        fig.update_traces(textposition='outside')
                
                # ===== 새로운 차트 유형들 =====
                
                # 도넛 차트
                elif "도넛" in prompt or "donut" in prompt.lower():
                    if mentioned_attrs and len(mentioned_attrs) >= 1:
                        figs = []
                        for attr in mentioned_attrs:
                            attr_df = df[df['속성'] == attr]
                            counts = attr_df['감성'].value_counts().reset_index()
                            counts.columns = ['감성', '리뷰수']
                            if len(counts) > 0:
                                fig_single = px.pie(counts, names='감성', values='리뷰수',
                                             title=f"📊 {attr} 감성 분포 (도넛)", hole=0.4,
                                             color='감성',
                                             color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                                fig_single.update_traces(textposition='inside', textinfo='label+value+percent')
                                figs.append(fig_single)
                        if figs:
                            fig = figs  # 리스트로 저장
                        else:
                            fig = None
                    else:
                        counts = plot_df['속성'].value_counts().reset_index()
                        counts.columns = ['속성', '리뷰수']
                        title = "📊 속성 분포 (도넛)"
                        fig = px.pie(counts, names=counts.columns[0], values='리뷰수', 
                                     title=title, hole=0.4,
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    # 리스트가 아닌 경우에만 update_traces 호출
                    if fig is not None and not isinstance(fig, list):
                        fig.update_traces(textposition='inside', textinfo='label+value+percent')
                    st.session_state.last_chart_type = "donut"
                
                # 가로 막대 차트
                elif "가로" in prompt or "horizontal" in prompt.lower():
                    counts = plot_df['속성'].value_counts().reset_index()
                    counts.columns = ['속성', '리뷰수']
                    if top_n:
                        counts = counts.head(top_n)
                    fig = px.bar(counts, y='속성', x='리뷰수', orientation='h',
                                 title="📊 속성 분포 (가로 막대)",
                                 text='리뷰수',
                                 color='속성',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_traces(textposition='outside')
                    st.session_state.last_chart_type = "horizontal"
                
                # 트리맵
                elif "트리맵" in prompt or "treemap" in prompt.lower():
                    counts = plot_df.groupby(['속성', '감성']).size().reset_index(name='리뷰수')
                    fig = px.treemap(counts, path=['속성', '감성'], values='리뷰수',
                                     title="📊 속성-감성 트리맵",
                                     color='리뷰수',
                                     color_continuous_scale='RdYlGn')
                    st.session_state.last_chart_type = "treemap"
                
                # 워드클라우드
                elif "워드" in prompt or "word" in prompt.lower() or "클라우드" in prompt:
                    if WORDCLOUD_AVAILABLE:
                        # 감성 필터 적용
                        wc_df = plot_df.copy()
                        if sentiment_filter:
                            wc_df = wc_df[wc_df['감성'] == sentiment_filter]
                        
                        # 리뷰 텍스트에서 워드클라우드 생성
                        text = " ".join(wc_df['리뷰'].dropna().astype(str).tolist())
                        
                        # 텍스트가 비어있으면 전체 데이터 사용
                        if not text.strip():
                            text = " ".join(df['리뷰'].dropna().astype(str).tolist())
                            st.info("ℹ️ 해당 조건의 리뷰가 없어 전체 리뷰로 워드클라우드를 생성합니다.")
                        
                        wordcloud_img_base64 = None
                        if text.strip():
                            # 한글 폰트 경로 (맥용)
                            font_paths = [
                                '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
                                '/Library/Fonts/AppleGothic.ttf',
                                '/System/Library/Fonts/AppleSDGothicNeo.ttc'
                            ]
                            font_path = None
                            for fp in font_paths:
                                import os
                                if os.path.exists(fp):
                                    font_path = fp
                                    break
                            
                            wordcloud = WordCloud(
                                font_path=font_path,
                                width=800, height=400,
                                background_color='white',
                                colormap='viridis'
                            ).generate(text)
                            
                            # matplotlib으로 워드클라우드 생성 및 base64 저장
                            import io
                            import base64
                            fig_wc, ax = plt.subplots(figsize=(10, 5))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            # 제목에 속성 및 감성 정보 표시
                            title_parts = []
                            if mentioned_attrs:
                                title_parts.append(', '.join(mentioned_attrs))
                            if sentiment_filter:
                                title_parts.append(sentiment_filter)
                            if title_parts:
                                ax.set_title(f"📊 {' '.join(title_parts)} 리뷰 워드클라우드")
                            else:
                                ax.set_title("📊 전체 리뷰 워드클라우드")
                            
                            # 이미지를 base64로 인코딩
                            buf = io.BytesIO()
                            fig_wc.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                            buf.seek(0)
                            wordcloud_img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                            plt.close()
                        else:
                            st.warning("⚠️ 워드클라우드를 생성할 리뷰 텍스트가 없습니다.")
                        fig = None  # plotly 차트 대신
                        st.session_state.last_chart_type = "wordcloud"
                    else:
                        st.warning("⚠️ 워드클라우드 기능을 사용하려면 `pip install wordcloud matplotlib` 명령어로 라이브러리를 설치해주세요.")
                        fig = None
                        wordcloud_img_base64 = None
                
                # 라인 차트 (속성별 감성 추이 - 막대 대신 라인으로)
                elif "라인" in prompt or "line" in prompt.lower() or "선" in prompt:
                    counts = plot_df.groupby(['속성', '감성']).size().reset_index(name='리뷰수')
                    fig = px.line(counts, x='속성', y='리뷰수', color='감성',
                                  title="📊 속성별 감성 분포 (라인)",
                                  markers=True,
                                  color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                    st.session_state.last_chart_type = "line"
                
                # ===== 기존 차트 유형들 =====
                
                # 파이차트 요청
                elif "파이" in prompt or "pie" in prompt.lower() or "원형" in prompt:
                    # 특정 속성이 선택된 경우 - 각 속성별로 개별 파이차트 생성
                    if mentioned_attrs and len(mentioned_attrs) >= 1:
                        figs = []  # 여러 차트 저장
                        for attr in mentioned_attrs:
                            attr_df = df[df['속성'] == attr]
                            counts = attr_df['감성'].value_counts().reset_index()
                            counts.columns = ['감성', '리뷰수']
                            if len(counts) > 0:
                                fig_single = px.pie(counts, names='감성', values='리뷰수', 
                                             title=f"📊 {attr} 감성 분포 (파이차트)",
                                             color='감성',
                                             color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                                fig_single.update_traces(textposition='inside', textinfo='label+value+percent')
                                figs.append(fig_single)
                        
                        # 모든 차트를 fig에 할당
                        if figs:
                            fig = figs  # 리스트로 저장
                        else:
                            st.warning(f"⚠️ 선택한 속성의 데이터가 없습니다.")
                            fig = None
                    elif "감성" in prompt:
                        # 전체 감성 분포 파이차트
                        counts = plot_df['감성'].value_counts().reset_index()
                        counts.columns = ['감성', '리뷰수']
                        fig = px.pie(counts, names='감성', values='리뷰수', 
                                     title="📊 감성 분포 (파이차트)",
                                     color='감성',
                                     color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                    else:
                        # 전체 속성 분포 파이차트 (기본)
                        counts = plot_df['속성'].value_counts().reset_index()
                        counts.columns = ['속성', '리뷰수']
                        if top_n:
                            counts = counts.head(top_n)
                        fig = px.pie(counts, names='속성', values='리뷰수', 
                                     title="📊 속성 분포 (파이차트)",
                                     color_discrete_sequence=px.colors.qualitative.Pastel)
                    # 리스트가 아닌 경우에만 update_traces 호출
                    if fig is not None and not isinstance(fig, list):
                        fig.update_traces(textposition='inside', textinfo='label+value+percent')
                    st.session_state.last_chart_type = "pie"
                # "속성별" 패턴 인식 - 감성 분포 차트
                elif ("속성별" in prompt.replace(" ", "") or "속성 별" in prompt) and ("차트" in prompt or "그래프" in prompt or "그려" in prompt):
                    # 속성별 감성 분포 (Grouped Bar Chart)
                    if top_n:
                        top_attrs = plot_df['속성'].value_counts().head(top_n).index.tolist()
                        filtered_df = plot_df[plot_df['속성'].isin(top_attrs)]
                        counts = filtered_df.groupby(['속성', '감성']).size().reset_index(name='리뷰수')
                        title = f"📊 속성별 감성 분포 (Top {top_n})"
                    else:
                        counts = plot_df.groupby(['속성', '감성']).size().reset_index(name='리뷰수')
                        title = "📊 속성별 감성 분포"
                    fig = px.bar(counts, x='속성', y='리뷰수', color='감성', 
                                 barmode='group',
                                 title=title,
                                 text='리뷰수',
                                 color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                    fig.update_traces(textposition='outside')
                    st.session_state.last_chart_type = "sentiment"
                elif "감성" in prompt and "속성" not in prompt:
                    # 전체 감성 분포
                    counts = plot_df['감성'].value_counts().reset_index()
                    counts.columns = ['감성', '리뷰수']
                    fig = px.pie(counts, names='감성', values='리뷰수', 
                                 title="📊 전체 감성 분포",
                                 color='감성',
                                 color_discrete_map={'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'})
                    fig.update_traces(textposition='inside', textinfo='label+value+percent')
                elif sentiment_filter and ("차트" in prompt or "그래프" in prompt or "비중" in prompt or "그려" in prompt):
                    # 특정 감성만 필터링하여 속성별 비중 차트
                    filtered_sentiment_df = plot_df[plot_df['감성'] == sentiment_filter]
                    counts = filtered_sentiment_df['속성'].value_counts().reset_index()
                    counts.columns = ['속성', '리뷰수']
                    total = counts['리뷰수'].sum()
                    counts['비중(%)'] = (counts['리뷰수'] / total * 100).round(1)
                    if top_n:
                        counts = counts.head(top_n)
                    
                    color_map = {'긍정': '#2ecc71', '부정': '#e74c3c', '중립': '#95a5a6'}
                    fig = px.bar(counts, x='속성', y='리뷰수', 
                                 title=f"📊 속성별 {sentiment_filter} 리뷰 비중",
                                 text=counts.apply(lambda x: f"{x['리뷰수']}건 ({x['비중(%)']}%)", axis=1),
                                 color_discrete_sequence=[color_map.get(sentiment_filter, '#3498db')])
                    fig.update_traces(textposition='outside')
                elif "차트" in prompt or "그래프" in prompt or "시각화" in prompt or "분포" in prompt or "그려" in prompt:
                    # 기본값: 속성별 분포 차트 (단순)
                    counts = plot_df['속성'].value_counts().reset_index()
                    counts.columns = ['속성', '리뷰수']
                    if top_n:
                        counts = counts.head(top_n)
                        title = f"📊 속성 분포 (Top {top_n})"
                    else:
                        title = "📊 속성 분포"
                    fig = px.bar(counts, x='속성', y='리뷰수', color='속성', 
                                 title=title,
                                 text='리뷰수',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_traces(textposition='outside')
                    st.session_state.last_chart_type = "simple"
                else:
                    # 차트 없음 (텍스트 질문)
                    fig = None
                
                # [Step 3.5] 키워드 검색 기능
                search_result = None
                search_keywords = ["찾아", "검색", "포함", "있는", "후기", "리뷰"]
                if any(kw in prompt for kw in search_keywords):
                    # 따옴표 안의 단어 추출 또는 일반 패턴
                    import re
                    # '단어' 또는 "단어" 패턴 찾기
                    quoted = re.findall(r"['\"]([^'\"]+)['\"]", prompt)
                    if quoted:
                        search_word = quoted[0]
                    else:
                        # "~가 들어간", "~를 찾아" 패턴에서 추출
                        words = prompt.replace("가", " ").replace("를", " ").replace("을", " ").replace("이", " ").split()
                        search_word = None
                        for i, w in enumerate(words):
                            if w in ["들어간", "포함된", "있는", "찾아", "검색"]:
                                if i > 0:
                                    search_word = words[i-1]
                                    break
                    
                    if search_word and len(search_word) > 1:
                        # 리뷰에서 검색
                        matched_reviews = df[df['리뷰'].str.contains(search_word, na=False)]
                        if len(matched_reviews) > 0:
                            search_result = f"\n\n**🔍 '{search_word}' 키워드 검색 결과 ({len(matched_reviews)}건)**\n\n"
                            for idx, row in matched_reviews.head(5).iterrows():
                                search_result += f"- [{row['속성']}] {row['리뷰'][:100]}...\n"
                            if len(matched_reviews) > 5:
                                search_result += f"\n... 외 {len(matched_reviews)-5}건 더 있습니다."
                
                # [Step 4] 리포트 작성
                status_text.write("✍️ 분석 리포트를 작성 중입니다...")
                
                # 속성별 리뷰수 준비 (항상 포함)
                attr_counts = df['속성'].value_counts().to_dict()
                sentiment_by_attr = df.groupby(['속성', '감성']).size().unstack(fill_value=0).to_dict()
                
                # 전체 데이터 요약
                full_data_summary = {
                    "총 리뷰 수": len(df),
                    "속성별 리뷰수": attr_counts,
                    "속성별 감성 분포": sentiment_by_attr
                }
                
                # 이전 대화 내역 가져오기 (최근 4개까지)
                conversation_history = ""
                if len(st.session_state.messages) > 0:
                    recent_msgs = st.session_state.messages[-4:]  # 최근 4개
                    for msg in recent_msgs:
                        role = "사용자" if msg["role"] == "user" else "AI"
                        conversation_history += f"{role}: {msg['content'][:200]}\n"
                
                # 리포트 생성 프롬프트 (대화 맥락 포함)
                system_prompt = """당신은 쇼핑 리뷰 데이터 분석 전문가입니다.
아래 데이터를 기반으로 사용자의 질문에 정확하게 답변하세요.

중요한 규칙:
1. 데이터에 있는 속성명을 정확히 사용하세요 (가격/가성비, 배송, 사용감, 기능/성능, 피부/안전, 포장, 냄새/향, 기타)
2. 이전 대화에서 언급된 주제가 있으면, 현재 질문에 주어가 불분명할 때 이전 주제를 참조하세요
3. 숫자는 정확하게 데이터에서 가져오세요
4. 차트/그래프는 시스템이 자동으로 생성하므로, 당신은 텍스트 분석과 인사이트만 제공하면 됩니다
5. 절대로 "텍스트 기반 환경에서는 차트를 생성할 수 없다"는 말을 하지 마세요
6. 리뷰 건수/개수는 사용자가 명시적으로 요청할 때만 제공하세요. 자동으로 리뷰 수를 언급하지 마세요
7. 그래프가 자동으로 생성된다는 점을 언급하지 마세요

이전 대화:
""" + conversation_history + """

전체 데이터 요약:
""" + str(full_data_summary)
                
                report_msg = [
                    ("system", system_prompt), 
                    ("user", f"질문: {prompt}")
                ]
                analysis_res = llm.invoke(report_msg).content
                
                # 키워드 검색 결과가 있으면 추가
                if search_result:
                    analysis_res += search_result
                
                # 완료 후 상태 메시지 제거
                status_text.empty()
                
                # 모든 작업 완료 시 상태 변경
                status.update(label="✅ 분석이 완료되었습니다!", state="complete", expanded=False)

            except Exception as e:
                status.update(label="❌ 분석 중 오류가 발생했습니다.", state="error")
                st.error(f"상세 에러: {e}")
                fig = None
                analysis_res = None

        # 최종 결과: 세션에 저장 (화면 출력은 상단 for loop에서 처리)
        if analysis_res is not None:
            # 차트에 사용된 속성 저장 (후속 질문에서 참조)
            chart_json = None
            wc_img = None
            
            if fig is not None:
                st.session_state.last_chart_attrs = plot_df['속성'].unique().tolist()
                # 차트를 JSON으로 변환하여 저장 (세션 직렬화 문제 방지)
                # 리스트인 경우 각각 변환
                if isinstance(fig, list):
                    chart_json = [f.to_dict() for f in fig]
                else:
                    chart_json = fig.to_dict()
            
            # 워드클라우드 이미지 저장 (있는 경우)
            if 'wordcloud_img_base64' in dir() and wordcloud_img_base64:
                wc_img = wordcloud_img_base64
                st.session_state.last_chart_attrs = plot_df['속성'].unique().tolist()
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": analysis_res, 
                "chart_json": chart_json,
                "wordcloud_img": wc_img
            })
            st.rerun()  # 세션 저장 후 페이지 새로고침하여 상단에서 출력

# 자동 스크롤 (Streamlit components 사용)
import streamlit.components.v1 as components

# 새 메시지가 있으면 맨 아래로 스크롤
if st.session_state.messages:
    components.html(
        """
        <script>
            // Streamlit의 부모 프레임으로 스크롤
            const streamlitDoc = window.parent.document;
            streamlitDoc.documentElement.scrollTop = streamlitDoc.documentElement.scrollHeight;
        </script>
        """,
        height=0
    )

# 스크롤바 항상 표시 CSS
st.markdown(
    """
    <style>
        /* 스크롤바 항상 표시 */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        ::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        ::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 5px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        html, body, [data-testid="stAppViewContainer"] {
            overflow-y: scroll !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)