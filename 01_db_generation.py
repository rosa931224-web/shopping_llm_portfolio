import os
import pandas as pd
import sqlite3
from dotenv import load_dotenv

# 1. 환경 설정
load_dotenv()

# 2. 경로 설정 (data 폴더 안의 원본 파일)
csv_file = 'data/09_reviews_long_for_tableau.csv'
db_file = 'shopping_reviews.db'

def create_review_db():
    try:
        # 데이터 로드
        df = pd.read_csv(csv_file)
        print(f"📂 [Step 01] 파일 로드 성공: {len(df)}건")

        # SQLite 연결 및 테이블 생성
        conn = sqlite3.connect(db_file)
        
        # 테이블명을 'naver_reviews'로 지정
        df.to_sql('naver_reviews', conn, if_exists='replace', index=False)
        
        print(f"✅ [Step 01] '{db_file}' 생성 및 데이터 저장 완료!")
        conn.close()
        
    except Exception as e:
        print(f"❌ [Step 01] 에러 발생: {e}")

if __name__ == "__main__":
    create_review_db()