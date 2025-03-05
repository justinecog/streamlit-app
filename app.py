import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
import os
import shutil
import pandas as pd
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai_tools import PDFSearchTool, DOCXSearchTool, TXTSearchTool, DirectoryReadTool

import openai

openai.api_key = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_MODEL_NAME"] = "o3-mini"

# 기본 폴더 경로
BASE_DIR = "dir"

# 최초 실행 시 1번만 폴더 생성 (세션 상태 유지)
if "foldername" not in st.session_state:
    current_time = datetime.now()
    st.session_state.foldername = current_time.isoformat().replace(":", "_")

UPLOAD_FOLDER = os.path.join(BASE_DIR, st.session_state.foldername)

# 폴더 생성 (최초 1회만 실행됨)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_uploaded_file(directory, file):
    file_path = os.path.join(directory, file.name)
    
    if file.name.endswith(".txt"):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(file.getvalue().decode("utf-8"))
    else:
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    return st.success(f"파일 업로드 성공! ({file.name})")

# 📂 폴더 삭제 함수
def delete_folder(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)  # 빈 폴더 다시 생성
        return st.success("📂 업로드된 파일이 모두 삭제되었습니다.")

# 📂 업로드된 파일 목록 조회 함수
def get_uploaded_files(directory):
    if os.path.exists(directory):
        return os.listdir(directory)
    return []

# 기본 형식
def main():
    st.title("회의록 작성 시스템2")

    # 📂 파일 업로드 섹션
    st.header("1️⃣ 파일 업로드")
    uploaded_file = st.file_uploader("파일을 업로드하세요 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        save_uploaded_file(UPLOAD_FOLDER, uploaded_file)

    # 📂 폴더 삭제 버튼 추가
    if st.button("📂 폴더 삭제"):
        delete_folder(UPLOAD_FOLDER)

    st.markdown("---")

    # 📋 업로드된 파일 목록 섹션
    st.header("2️⃣ 업로드된 파일 목록")
    files = get_uploaded_files(UPLOAD_FOLDER)

    if len(files) == 0:
        st.warning("현재 업로드된 파일이 없습니다.")
    else:
        st.success(f"현재 저장된 파일 수: {len(files)}개")
        file_df = pd.DataFrame({"파일명": files}, index=range(1, len(files) + 1))
        st.table(file_df)

    st.markdown("---")

    # 📝 회의록 작성 섹션
    st.header("3️⃣ 회의록 작성")
    
    meeting_name = st.text_input("회의 이름을 입력하세요:", "")
    meeting_topic = st.text_input("회의 주제를 입력하세요:", "")
    
    start_button = st.button("회의록 작성 시작")

    output_placeholder = st.empty()  # CrewAI 결과 출력을 위한 placeholder

    if start_button:
        if not meeting_name.strip():
            st.warning("⚠️ 회의 이름을 입력하세요!")
            return
        
        meeting_name = meeting_name.strip()
        
        if not meeting_topic.strip():
            st.warning("⚠️ 회의 주제를 입력하세요!")
            return

        meeting_topic = meeting_topic.strip()
        
        log_text = f"🔹 '{meeting_topic}' 주제에 대한 회의록 작성을 시작합니다...\n"
        output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)

        if not files:
            log_text += "⚠️ 업로드된 파일이 없습니다. 먼저 파일을 업로드하세요.\n"
            output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)
            return

        log_text += f"📄 총 {len(files)}개의 파일을 분석 중...\n"
        output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)

        # CrewAI 설정
        pdf_search_tool = PDFSearchTool()
        docx_search_tool = DOCXSearchTool()
        txt_search_tool = TXTSearchTool()
        directory_read_tool = DirectoryReadTool(directory=UPLOAD_FOLDER)

        researcher = Agent(
            role="시니어 컨설턴트 겸 리서처",
            goal="문서에서 검색해서 분석 후 리서치 정리",
            backstory="""
            다수의 정보를 활용하여 항상 최고의 리서치 결과를 만드는 시니어 리서처. 출처가 사실인지 체크하고 구체적인 예와 통찰을 포함.
            """,
            allow_delegation=False,
            verbose=True,
            tools = [
                directory_read_tool,
                pdf_search_tool,
                docx_search_tool,
                txt_search_tool
            ],
        )
    
        editor = Agent(
            role="전문적인 에디터",
            goal="회의록을 명확하고 체계적으로 정리하여 최종 문서로 완성",
            backstory="""
            다년간 전문 문서 편집 경험을 가진 에디터. 논리적 흐름을 정리하고, 문장의 가독성을 높이며, 중요한 정보를 빠짐없이 포함하여 명확한 문서를 작성하는 것이 목표.
            회의록에서 핵심 논의를 요약하고, 의사 결정 사항과 후속 조치를 구조적으로 정리함.
            전문적인 문서 작성 스타일을 유지하면서도, 읽기 쉬운 형태로 문장을 다듬음.
            """,
            allow_delegation=False,
            verbose=True,
        )

        # researcher와 editor가 수행할 작업 정의
        research_task = Task(
            agent=researcher,
            description="""'{topic}'에 대한 정보를 수집하고 분석하세요. 당신은 철저한 연구를 수행하는 AI입니다.
            주어지는 파일 (pdf, docx, txt) 모두를 분석해주세요.
        당신의 응답은 검증된 출처, 데이터 또는 공식 문서를 기반으로 해야 합니다.  
        - 출처를 반드시 제공하세요. 출처가 없다면, "출처를 찾을 수 없습니다."라고 답변하세요.  
        - 신뢰할 수 없는 정보는 포함하지 마세요.  
        - 모호하거나 확인되지 않은 내용은 절대 생성하지 마세요.  
        """,
            expected_output="'{topic}'에 대한 정보를 수집하고 분석해서 정리해줘."
        )
        
        edit_task = Task(
            agent= editor,
            description="""'{topic}'에 대한 회의록을 작성해주세요. 
            연구 결과를 검토하고 다듬어 완성된 문서로 만들어주세요.  
        당신의 응답은 검증된 출처, 데이터 또는 공식 문서를 기반으로 해야 합니다.  
        - 출처를 반드시 제공하세요. 출처가 없다면, "출처를 찾을 수 없습니다."라고 답변하세요.  
        - 신뢰할 수 없는 정보는 포함하지 마세요.  
        - 모호하거나 확인되지 않은 내용은 절대 생성하지 마세요.
        """,
            expected_output="'{meeting_name}'이 회의 제목이고 '{topic}'이 회의 주제인 회의록을 작성해 줘.",
            #output_file=f"회의록_{meeting_topic}.txt",
            depends_on=[research_task] 
        )
        crew = Crew(
            agents=[researcher, editor],
            tasks=[research_task, edit_task],
            process=Process.sequential,
            verbose=True,
        )

        # ✅ CrewAI 실행 후 결과를 실시간 업데이트
        log_text += "🔍 AI 분석 시작 (1-2분 정도 걸립니다)...  \n"
        output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)

        result = crew.kickoff(inputs={"topic": meeting_topic, "meeting_name": meeting_name})  # CrewAI 실행

        # CrewAI 결과 업데이트
        log_text += f"\n🔹 CrewAI 결과:\n{str(result)}\n"
        output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)

        # CrewAI 결과를 파일로 저장
        output_file_path = os.path.join("./", f"회의록_{meeting_name}.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(str(result))  # ✅ CrewOutput을 문자열로 변환하여 저장

        log_text += f"✅ '{meeting_topic}' 주제에 대한 회의록 작성을 완료하였습니다!\n"
        output_placeholder.text_area("📜 진행 상황 및 결과", log_text, height=300)

        # 파일 다운로드 버튼 제공
        with open(output_file_path, "rb") as file:
            st.download_button("📥 회의록 다운로드", file, file_name=f"회의록_{meeting_name}.txt")

if __name__ == "__main__":
    main()
