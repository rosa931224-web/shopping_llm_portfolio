import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# 1. 환경 변수 로드 (.env에 있는 API 키 등을 불러올 준비)
load_dotenv()

# 2. 데이터 불러오기 
# 파일이 shopping_llm_portfolio 폴더 바로 안에 있어야 합니다.
csv_file = 'data/09_reviews_long_for_tableau.csv'

try:
    df = pd.read_csv(csv_file)
    print(f"📂 파일 로드 성공! 총 {len(df)}건의 리뷰를 확인했습니다.")

    # 3. SQLite 데이터베이스 연결 (파일이 없으면 자동으로 생성됩니다)
    # 이름은 지윤님 프로젝트에 맞춰 shopping_reviews.db로 정했습니다.
    conn = sqlite3.connect('shopping_reviews.db')

    # 4. 데이터를 'naver_reviews' 테이블로 저장
    # index=False는 데이터프레임의 인덱스 번호를 DB에 넣지 않겠다는 뜻입니다.
    df.to_sql('naver_reviews', conn, if_exists='replace', index=False)
    
    print("✅ 'shopping_reviews.db' 생성이 완료되었습니다!")
    print("🖥️ 이제 'naver_reviews' 테이블에서 AI가 데이터를 꺼내올 수 있습니다.")

    conn.close()

except FileNotFoundError:
    print(f"❌ 에러: '{csv_file}' 파일을 찾을 수 없습니다. 폴더 위치를 확인해 주세요.")
except Exception as e:
    print(f"❌ 예상치 못한 에러 발생: {e}")