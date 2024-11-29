import gradio as gr
import fitz  # PyMuPDF
import re
from typing import List, Tuple
from transformers import pipeline, NllbTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import numpy as np
import langdetect

# 전역 변수로 PDFProcessor 인스턴스 생성
pdf_processor = None

class PDFProcessor:
    def __init__(self):
        self.sections = []
        self.title = None
        print("요약 모델 로딩 중...")
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        print("임베딩 모델 로딩 중...")
        self.embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("번역 모델 로딩 중...")
        self.translator = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        self.translator_tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        self.sentences = []
        self.sentence_embeddings = None
        self.is_english_pdf = None
        print("모든 모델 로딩 완료!")

    def _is_excluded_text(self, text: str) -> bool:
        """Figure/Table 등 제외할 텍스트인지 확인"""
        if re.match(r'^\s*(?:Fig(?:ure)?|Table)\s*\d+[\s\.]', text, re.IGNORECASE):
            return True
        if re.match(r'^\s*(?:Fig(?:ure)?|Table)\s*\d+.*(?:\.|\?|\!|:)$', text, re.IGNORECASE):
            return True
        if any(text.lower().startswith(start) for start in [
            'fig.', 'figure', 'table', 'supplementary fig', 
            'supplementary table', 'supplementary figure',
            'references', 'acknowledgments', 'funding'
        ]):
            return True
        return False

    def detect_language(self, text: str) -> bool:
        """텍스트의 언어가 영어인지 감지"""
        try:
            lang = langdetect.detect(text)
            return lang == 'en'
        except:
            return False

    def translate_text(self, text: str, from_lang: str, to_lang: str) -> str:
        """텍스트 번역"""
        try:
            # 특수문자 및 줄바꿈 정리
            text = self.clean_text(text)
            
            # 영어 단어/약어 패턴
            eng_pattern = r'[A-Z][A-Za-z]*(?:-[A-Za-z]+)*|[A-Z]{2,}(?:-[A-Z]+)*'
            
            # 전문용어/약어 찾기
            special_terms = set()
            matches = re.finditer(eng_pattern, text)
            for match in matches:
                word = match.group()
                if (len(word) >= 2 and (word.isupper() or '-' in word or 
                    (word[0].isupper() and any(c.isupper() for c in word[1:])))):
                    special_terms.add(word)
            
            # 긴 단어부터 처리 (부분 매칭 방지)
            modified_text = text
            for term in sorted(special_terms, key=len, reverse=True):
                modified_text = modified_text.replace(term, f"'{term}'")
            
            # NLLB 번역
            lang_map = {'ko': 'kor_Hang', 'en': 'eng_Latn'}
            from_code = lang_map[from_lang]
            to_code = lang_map[to_lang]
            
            inputs = self.translator_tokenizer(
                modified_text,
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            translated = self.translator.generate(
                **inputs,
                forced_bos_token_id=self.translator_tokenizer.convert_tokens_to_ids(to_code),
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                early_stopping=True
            )
            
            result = self.translator_tokenizer.batch_decode(
                translated, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            # 결과 정리
            result = result.replace("'", "").replace('"', "")
            result = re.sub(r'\s+', ' ', result).strip()
            
            print(f"번역 - 원문: {text}")
            print(f"번역 - 수정된 입력: {modified_text}")
            print(f"번역 - 결과: {result}")
            
            return result
        except Exception as e:
            print(f"번역 중 오류 발생: {str(e)}")
            return text

    def get_bookmarks_and_content(self, pdf_path: str) -> Tuple[str, List[Tuple[str, str]]]:
        """PDF의 책갈피 정보 추출"""
        doc = fitz.open(pdf_path)
        try:
            toc = doc.get_toc(simple=False)
            if toc:
                print("책갈피를 찾았습니다. 책갈피 기반으로 처리합니다.")
                return self._process_with_bookmarks(doc, toc)
            print("책갈피가 없습니다. 텍스트 패턴 기반으로 처리합니다.")
            return self._process_without_bookmarks(doc)
        finally:
            doc.close()

    def clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 줄바꿈 및 하이픈으로 나뉜 단어 처리
        text = re.sub(r'-\n', '', text)
        text = re.sub(r'\n', ' ', text)
        # 여러 개의 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def prepare_qa_embeddings(self, doc):
        """QA를 위한 전체 PDF 내용의 임베딩 준비"""
        print("\n전체 문장 추출 중...")
        self.sentences = []
        
        # 첫 페이지로 PDF 언어 감지
        first_page_text = doc[0].get_text()
        self.is_english_pdf = self.detect_language(first_page_text)
        print(f"PDF 언어: {'영어' if self.is_english_pdf else '한국어'}")
        
        for page in doc:
            text = page.get_text()
            # 텍스트 정리
            text = self.clean_text(text)
            sentences = text.split('. ')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 30 and not self._is_excluded_text(sentence):
                    self.sentences.append(sentence)
        
        print(f"총 {len(self.sentences)}개 문장 추출됨")
        print("문장 임베딩 생성 중...")
        self.sentence_embeddings = self.embedding_model.encode(self.sentences)
        print("임베딩 생성 완료!")

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """질문에 대한 답변 찾기"""
        if not self.sentences or self.sentence_embeddings is None:
            return "PDF를 먼저 처리해주세요."

        try:
            if self.is_english_pdf:
                translated_question = self.translate_text(question, 'ko', 'en')
                print(f"번역된 질문: {translated_question}")
            else:
                translated_question = question

            question_embedding = self.embedding_model.encode([translated_question])
            similarities = np.dot(self.sentence_embeddings, question_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            result = "관련 내용:\n\n"
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity > 0.3:
                    content = self.clean_text(self.sentences[idx])
                    if self.is_english_pdf:
                        translated_content = self.translate_text(content, 'en', 'ko')
                        result += f"[유사도: {similarity:.3f}]\n"
                        result += f"원문: {content}\n"
                        if translated_content and translated_content != content:
                            result += f"번역: {translated_content}\n"
                        result += "\n"
                    else:
                        result += f"[유사도: {similarity:.3f}]\n"
                        result += f"{content}\n\n"
            
            return result if len(result) > 20 else "관련된 내용을 찾을 수 없습니다."
        except Exception as e:
            print(f"답변 생성 중 오류 발생: {str(e)}")
            return "답변 생성 중 오류가 발생했습니다."

    def _process_with_bookmarks(self, doc, toc) -> Tuple[str, List[Tuple[str, str]]]:
        """책갈피 기반 처리"""
        try:
            title = None
            main_sections = []
            
            # 제외할 섹션 패턴
            exclude_patterns = [
                r'credit.*statement',
                r'declaration.*interest',
                r'data availability',
                r'acknowledgments?',
                r'references',
                r'appendix',
                r'bibliography'
            ]
            
            # 섹션 정보 수집
            section_info = []
            for i, item in enumerate(toc):
                try:
                    level, text, page, info = item
                    
                    if level == 1 and title is None:
                        title = text
                        continue
                    
                    if level in [1, 2]:
                        if not any(re.search(pattern, text.lower()) for pattern in exclude_patterns):
                            if 0 <= page-1 < len(doc):  # 페이지 번호 유효성 검사
                                section_info.append({
                                    'full_name': text,
                                    'page': page,
                                    'info': info
                                })
                except Exception as e:
                    print(f"섹션 정보 처리 중 오류: {str(e)}")
                    continue
            
            # 각 섹션의 내용 추출
            for i, section in enumerate(section_info):
                try:
                    current_page = doc[section['page']-1]
                    content = current_page.get_text()
                    
                    # 내용이 너무 짧으면 다음 페이지도 포함
                    if len(content.split()) < 50 and section['page'] < len(doc):
                        content += doc[section['page']].get_text()
                    
                    if content.strip():
                        # 텍스트 길이에 따라 요약 길이 조정
                        word_count = len(content.split())
                        if word_count < 50:
                            main_sections.append((section['full_name'], content.strip()))
                        else:
                            try:
                                max_length = min(word_count // 3, 150)
                                min_length = min(max_length // 2, 30)
                                
                                summary = self.summarizer(
                                    content.strip()[:1024],  # BART 모델 입력 제한
                                    max_length=max_length,
                                    min_length=min_length,
                                    do_sample=False
                                )[0]['summary_text']
                                main_sections.append((section['full_name'], summary))
                            except Exception as e:
                                print(f"요약 중 오류 발생: {str(e)}")
                                # 오류 시 원문의 일부만 사용
                                main_sections.append((section['full_name'], content[:500] + "..."))
                except Exception as e:
                    print(f"섹션 '{section['full_name']}' 처리 중 오류: {str(e)}")
                    continue
            
            if not main_sections:
                # 섹션 처리 실패 시 전체 텍스트를 하나의 섹션으로 처리
                return self._process_without_bookmarks(doc)
            
            return title or "제목 없음", main_sections
        
        except Exception as e:
            print(f"책갈피 처리 중 오류 발생: {str(e)}")
            return self._process_without_bookmarks(doc)

    def _process_without_bookmarks(self, doc) -> Tuple[str, List[Tuple[str, str]]]:
        """책갈피가 없는 경우의 처리"""
        try:
            title = "제목 없음"
            main_sections = []
            
            # 전체 텍스트 수집
            full_text = ""
            for page in doc:
                try:
                    text = page.get_text()
                    if text.strip():
                        full_text += text + "\n"
                except Exception as e:
                    print(f"페이지 텍스트 추출 중 오류: {str(e)}")
                    continue

            def summarize_chunk(text: str, chunk_num: int) -> str:
                """��크 요약 함수"""
                try:
                    # 입력 텍스트가 너무 짧으면 그대로 반환
                    if len(text.split()) < 50:
                        return text

                    # BART 모델을 위한 텍스트 전처리
                    # 입력 텍스트를 200-400 단어로 제한
                    words = text.split()
                    if len(words) > 400:
                        text = " ".join(words[:400])
                    
                    # 요약 길이 설정
                    max_length = min(len(text.split()) // 2, 130)
                    min_length = min(max_length // 2, 30)

                    summary = self.summarizer(
                        text,
                        max_length=max_length,
                        min_length=min_length,
                        do_sample=False,
                        truncation=True
                    )[0]['summary_text']
                    
                    return summary
                except Exception as e:
                    print(f"청크 {chunk_num} 요약 중 오류: {str(e)}")
                    # 오류 시 원문의 처음 200단어 반환
                    return " ".join(text.split()[:200]) + "..."

            # 텍스트를 더 작은 청크로 분할 (약 300단어 단위)
            words = full_text.split()
            chunk_size = 300
            chunks = []
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():  # 빈 청크 제외
                    chunks.append(chunk)

            # 청크가 없으면 전체 텍스트를 하나의 청크로
            if not chunks:
                chunks = [full_text]

            # 각 청크 처리
            for i, chunk in enumerate(chunks, 1):
                summary = summarize_chunk(chunk, i)
                section_title = f"섹션 {i}" if len(chunks) > 1 else "전체 내용"
                main_sections.append((section_title, summary))

            return title, main_sections

        except Exception as e:
            print(f"PDF 처리 중 오류 발생: {str(e)}")
            return "처리 오류", [("오류", "PDF를 처리할 수 없습니다.")]

def process_pdf_and_prepare_qa(pdf_file) -> Tuple[str, str, str]:
    """PDF 리: 요약 및 QA 준비를 동시에 수행"""
    if pdf_file is None:
        return "", "", ""
    
    try:
        print("\nPDF 처리 시작...")
        doc = fitz.open(pdf_file.name)
        
        # QA를 위한 임베딩 준비
        pdf_processor.prepare_qa_embeddings(doc)
        
        # 요약 생성
        title, sections = pdf_processor.get_bookmarks_and_content(pdf_file.name)
        pdf_processor.sections = sections
        
        result = f"제목: {title}\n\n"
        for section_title, summary in sections:
            result += f"=== {section_title} ===\n"
            result += f"{summary}\n\n"
        
        doc.close()
        print("PDF 처리 완료!")
        return result, "", ""  # summary_output, question_input, answer_output
        
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {str(e)}")
        return "PDF 처리 중 오류가 발생했습니다.", "", ""

def answer_question(pdf_file, question):
    """질문에 대한 답변 처리"""
    global pdf_processor
    if not question.strip():
        return "질문을 입력해주세요."
    if not pdf_processor.sentences:
        return "PDF를 먼저 업로드하고 '요약하기' 버튼을 눌러주세요."
    return pdf_processor.answer_question(question)

def create_interface():
    global pdf_processor
    pdf_processor = PDFProcessor()
    
    with gr.Blocks() as demo:
        gr.Markdown("# PDF 요약 및 QA")
        
        with gr.Row():
            pdf_file = gr.File(label="PDF 파일을 업로드하세요", type="filepath")
        
        with gr.Row():
            with gr.Column():
                summary_output = gr.Textbox(label="PDF 요약", lines=20)
            
            with gr.Column():
                question_input = gr.Textbox(label="질문을 입력하세요", lines=2)
                answer_button = gr.Button("질문하기")
                answer_output = gr.Textbox(label="질문 답변", lines=10)
        
        # PDF 파일 변경 이벤트 처리
        pdf_file.change(
            fn=process_pdf_and_prepare_qa,  # 요약과 QA 준비를 동시에 처리
            inputs=[pdf_file],
            outputs=[summary_output, question_input, answer_output]
        )
        
        answer_button.click(
            fn=answer_question,
            inputs=[pdf_file, question_input],
            outputs=[answer_output]
        )
        
        return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
