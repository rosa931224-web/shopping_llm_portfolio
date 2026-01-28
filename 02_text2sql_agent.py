import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langgraph.checkpoint.memory import MemorySaver # 👈 기억력을 위한 도구

# 1. 환경 변수 로드
load_dotenv()
my_key = os.getenv("GOOGLE_API_KEY")

# 2. DB 연결
db = SQLDatabase.from_uri("sqlite:///shopping_reviews.db")

# 3. LLM 설정 (2026년 최신 모델)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=my_key, 
    temperature=0
)

# 4. 메모리 설정 (대화 내용을 저장하는 수첩)
memory = MemorySaver()

# 5. 랭그래프 기반 SQL 에이전트 생성
# 'checkpointer'를 넣으면 대화 맥락이 유지됩니다.
agent_executor = create_sql_agent(
    llm, 
    db=db, 
    agent_type="openai-tools", 
    verbose=True,
    checkpointer=memory
)

# 6. 연속 대화 테스트
# 'thread_id'는 지윤님과의 대화방 고유 번호예요.
config = {"configurable": {"thread_id": "jiyun_session_001"}}

print("\n--- 🤖 랭그래프 분석 비서 가동 (기억력 탑재) ---")

def ask_question(query):
    print(f"\n🙋 지윤: {query}")
    try:
        response = agent_executor.invoke({"input": query}, config=config)
        print(f"🤖 비서: {response['output']}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

# 연속 질문 테스트!
ask_question("배송 관련 리뷰는 총 몇 건이야?")
ask_question("그중에서 '부정' 리뷰는 몇 건이야?") 
ask_question("그중에서 제일 많이 언급 된 키워드 3개만 알려줘")