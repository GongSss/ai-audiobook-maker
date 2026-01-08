import streamlit as st
import os
import json
import time
import re
import wave
import difflib
import contextlib
import io
import base64
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from pydub import AudioSegment

# ==========================================
# [0] 기본 설정 및 상수
# ==========================================
SCRIPT_ROOT = "playground_scripts"
AUDIO_ROOT = "playground_audio"
REMAINING_FILE = "remaining_source.txt"

# 모델 설정
TTS_MODEL_ID = "gemini-2.5-pro-preview-tts" 
VERIFY_MODEL_ID = "gemini-2.5-pro"
ANALYZE_MODEL_ID = "gemini-3-pro-preview"

# 보이스 목록
VOICE_OPTIONS = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Zephyr"]

os.makedirs(SCRIPT_ROOT, exist_ok=True)
os.makedirs(AUDIO_ROOT, exist_ok=True)

# ==========================================
# [1] 유틸리티: 텍스트 처리
# ==========================================
def save_text_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def load_text_file(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

def get_full_text_from_epub(file):
    """
    EPUB의 Spine(읽는 순서)을 기반으로 정확한 순서대로 텍스트를 추출합니다.
    """
    # 임시 파일로 저장
    with open("temp.epub", "wb") as f:
        f.write(file.getbuffer())
        
    full_text = []
    
    try:
        book = epub.read_epub("temp.epub")
        
        # [핵심 수정] get_items() 대신 spine을 순회합니다.
        # book.spine은 (item_id, linear) 튜플의 리스트입니다.
        for item_id, _ in book.spine:
            item = book.get_item_with_id(item_id)
            
            # 아이템이 존재하고, 문서(HTML) 타입인 경우에만 처리
            if item and item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # [추가 팁] Script나 Style 태그는 텍스트가 아니므로 제거 (더 깔끔해짐)
                for script in soup(["script", "style"]):
                    script.extract()
                    
                text = soup.get_text(separator='\n').strip()
                
                if text:
                    full_text.append(text)
                    
        return "\n\n".join(full_text)
        
    except Exception as e:
        print(f"Epub Extract Error: {e}")
        return ""
        
    finally:
        if os.path.exists("temp.epub"):
            try:
                os.remove("temp.epub")
            except:
                pass

def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = name.replace(" ", "_")
    return name.strip()[:50]

def get_subdirectories(root_path):
    if not os.path.exists(root_path): return []
    return sorted([d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])

def get_files_in_dir(dir_path, ext):
    if not os.path.exists(dir_path): return []
    return sorted([f for f in os.listdir(dir_path) if f.endswith(ext)])

def process_text_for_playground(text, max_chars=1600):
    """
    텍스트 전처리:
    1. 불필요한 괄호 및 내용 삭제 (주석, 지문 등)
    2. 공백 정리
    3. 문장 단위로 끊어서 max_chars 길이에 맞게 청크 분할
    """
    # [NEW] 1. 괄호와 그 안의 내용 통째로 삭제
    # 예: "안녕하세요(웃음)" -> "안녕하세요"
    text = re.sub(r'\([^)]*\)', '', text)   # (소괄호) 제거
    text = re.sub(r'\[[^\]]*\]', '', text)  # [대괄호] 제거
    text = re.sub(r'\{[^}]*\}', '', text)   # {중괄호} 제거
    text = re.sub(r'\<[^>]*\>', '', text)   # <꺽쇠> 제거
    
    # 2. 특수문자 및 공백 정리
    # 줄바꿈을 공백으로 변경
    text = text.replace("\n", " ")
    # 말줄임표 정규화
    text = text.replace("...", ",").replace("…", ",")
    # 남은 특수문자 제거 (필요하다면 * # @ 등)
    text = re.sub(r'[\*\#\@]', '', text)
    
    # 다중 공백을 하나로 줄임 ("  " -> " ")
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. 청크 분할 (기존 로직 유지)
    chunks = []
    current_chunk = ""
    # 마침표, 물음표, 느낌표 뒤에서 문장 분리
    sentences = re.split(r'(?<=[.?!])\s+', text)
    
    for sentence in sentences:
        if not sentence: continue
        
        # 청크 크기 제한 확인
        if len(current_chunk) + len(sentence) > max_chars:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk: current_chunk += " " + sentence
            else: current_chunk = sentence
                
    if current_chunk: chunks.append(current_chunk.strip())
    return chunks


# ==========================================
# [1-2] 유틸리티: 타임라인 자동 보정 (삭제 시)
# ==========================================
def adjust_timeline_for_deletion(timeline, del_start, del_end):
    gap = del_end - del_start
    new_timeline = []
    
    for item in timeline:
        # 1. 삭제 구간보다 앞에 있는 문장 -> 영향 없음
        if item['end'] <= del_start:
            new_timeline.append(item)
            
        # 2. 삭제 구간보다 뒤에 있는 문장 -> 시간 당김 (Gap만큼 빼기)
        elif item['start'] >= del_end - 0.05: # 0.05는 미세 오차 허용
            item['start'] = max(0, item['start'] - gap)
            item['end'] = max(0, item['end'] - gap)
            new_timeline.append(item)
            
        # 3. 삭제 구간에 걸쳐 있는 문장 -> 길이 줄임
        else:
            # 뒷부분이 잘려나갔으므로 끝나는 시간만 당김
            # (단, 시작 시간은 그대로)
            item['end'] = max(item['start'], item['end'] - gap)
            new_timeline.append(item)
            
    return new_timeline

# ==========================================
# [1-3] 유틸리티: 타임라인 자동 보정 (재녹음/패칭 시)
# ==========================================
def adjust_timeline_for_patch(timeline, target_start, target_end, new_duration):
    """
    target_start ~ target_end 구간이 new_duration 길이로 바뀌었을 때,
    뒷부분을 밀거나 당깁니다.
    """
    old_duration = target_end - target_start
    diff = new_duration - old_duration # 양수면 밀리고, 음수면 당겨짐
    
    # 오차 범위 (재녹음된 구간의 끝 지점)
    patch_end_point = target_start + new_duration
    
    for item in timeline:
        # 1. 재녹음 구간 이후의 문장들 -> 차이(diff)만큼 이동
        if item['start'] >= target_end - 0.1:
            item['start'] += diff
            item['end'] += diff
            
        # 2. 재녹음된 바로 그 문장 (또는 그 안에 포함된 문장)
        elif item['end'] > target_start and item['start'] < target_end:
            # 이 문장의 끝을 새로운 오디오 끝에 맞춤 (단순화)
            # 여러 문장이 겹친 경우 복잡하지만, 보통 1문장 수선이므로 이 방식이 유효함
            item['end'] += diff 
            
    return timeline

# ==========================================
# [2] 유틸리티: 오디오 생성 & 처리 (Raw PCM 대응 패치)
# ==========================================

def add_silence_padding(audio_bytes, duration_sec=0.5):
    """
    [최종 수정] Raw PCM 대응 강화
    - RIFF 헤더가 있으면 WAV로 인식
    - 헤더가 없고 0000으로 시작해도, 데이터가 충분히 크면 Raw PCM으로 처리하여 살려냄
    """
    # 1. 1차 방어: 데이터가 너무 작으면(100바이트 미만) 진짜 오류
    if not audio_bytes or len(audio_bytes) < 100:
        print("Padding Error: 데이터가 비어있습니다.")
        return None

    # Gemini 기본 오디오 스펙 (Raw PCM일 경우 적용할 값)
    # 모델에 따라 24000Hz가 기본인 경우가 많음
    CHANNELS = 1
    SAMPWIDTH = 2
    FRAMERATE = 24000
    
    output_io = io.BytesIO()
    
    try:
        raw_data = audio_bytes
        
        # 2. 데이터 형식 판단 (WAV vs Raw PCM)
        if audio_bytes.startswith(b'RIFF'):
            # WAV 파일인 경우: 헤더 정보를 읽어서 세팅
            try:
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_in:
                    CHANNELS = wav_in.getnchannels()
                    SAMPWIDTH = wav_in.getsampwidth()
                    FRAMERATE = wav_in.getframerate()
                    raw_data = wav_in.readframes(wav_in.getnframes())
            except wave.Error:
                print("Warning: RIFF 헤더가 있지만 wave 모듈이 읽지 못함. Raw로 간주합니다.")
        else:
            # WAV가 아닌 경우 (RIFF 없음):
            # 00 00 00 00으로 시작하더라도 Raw PCM으로 간주하고 진행
            # 별도 처리 없이 raw_data = audio_bytes 그대로 사용
            pass

        # 3. 무음 데이터 생성
        num_silent_frames = int(FRAMERATE * duration_sec)
        silent_data = b'\x00' * (num_silent_frames * CHANNELS * SAMPWIDTH)

        # 4. 데이터 결합 (무음 + 원본)
        final_data = silent_data + raw_data

        # 5. 새로운 WAV 컨테이너 씌우기 (Re-packaging)
        with wave.open(output_io, 'wb') as wav_out:
            wav_out.setnchannels(CHANNELS)
            wav_out.setsampwidth(SAMPWIDTH)
            wav_out.setframerate(FRAMERATE)
            wav_out.writeframes(final_data)
            
        return output_io.getvalue()

    except Exception as e:
        print(f"Padding Critical Error: {e}")
        return None

def generate_speech(api_key, text, prompt, voice_name, temperature, speed_rate=1.0, volume_db=0.0):
    """
    [수정됨] volume_db 파라미터 포함 버전
    """
    client = genai.Client(api_key=api_key)
    
    # 가이드라인 (기존 내용 유지)
    audio_guidelines = """
    [Audio Engineering Guidelines - STRICTLY FOLLOW]:
    1. Tone: 자연스러운 일시 정지를 포함한 부드럽고 차분하며 듣기 편안한 책 읽기 톤을 유지합니다.
    2. Pitch: 0.0으로 설정하며, 처음부터 끝까지 일관되게 유지합니다.
    3. Speaking Rate: -1.0 (Slow/Steady)으로 설정하며, 처음부터 끝까지 느리고 꾸준하며 균일하게 유지합니다.
    4. Volume: 처음부터 끝까지 일관되고 균형 있게 유지합니다.
    5. Initial Word Intonation: 첫 단어 시작 시 강한 억양을 피하고, 평탄하고 안정적인 시작을 유지합니다.
    6. Emphasis: 특정 단어에 대한 강조 없이 평탄하고 균일하게 전달합니다.
    7. Pauses between sentences: 짧고 자연스러운 일시 정지만 허용하며, 긴 일시 정지나 들리는 숨소리는 피합니다.
    8. Breathing Sounds / Breath Control: 숨소리는 완전히 제거되도록 최소화하며 들리지 않아야 합니다. 문장 중 또는 문장 사이에서 숨소리가 없어야 하며, 최종 문장의 끝까지 무음이 유지되어야 합니다.
    9. Sibilance Suppression / Consonant Handling: ㅅ, ㅆ, ㅊ, ㅈ, ㅍ 등 마찰음은 날카롭거나 거칠게 들리지 않도록 부드럽고 유연하게 발음합니다 (완전 억제).
    10. Consistency: 처음부터 마지막 문장까지, 그리고 모든 재생성 과정에 걸쳐 동일한 Pitch, Tone, Speed, 말하는 스타일을 유지해야 합니다.
    11. Naturalness: 주저함, 더듬거림, 왜곡, Robotic artifacts 없이 부드럽고 연속적으로 읽습니다.
    12. Audio quality: 깨끗하고 명확하며 방송 품질 수준의 결과물을 제공합니다.
    """

    system_prompt = (
        "You are a professional AI Audio Book Narrator and Mastering Engineer. "
        "Your goal is to generate high-fidelity speech that sounds exactly like a professional broadcast recording."
        "DO NOT generate any text response. JUST generate the AUDIO output."
    )
    
    full_prompt = f"""
    {system_prompt}

    {audio_guidelines}

    [User Style Instruction]:
    {prompt}

    [Speed Control]:
    The base speaking rate is set to '-1.0' (Slow) as per guidelines.
    Current Adjustment: Please apply a speed multiplier of {speed_rate}x to this base rate.

    [Text to Read]:
    {text}
    """
    
    config = types.GenerateContentConfig(
        temperature=temperature,
        response_modalities=["AUDIO"], 
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )
    )

    try:
        # 1. AI 오디오 생성 요청
        response = client.models.generate_content(
            model=TTS_MODEL_ID,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=full_prompt)])],
            config=config
        )
        
        raw_audio = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                raw_audio = part.inline_data.data
                break
        
        # 2. 볼륨 조절 로직 (pydub 사용)
        if raw_audio:
            if volume_db != 0.0:
                try:
                    s = AudioSegment.from_wav(io.BytesIO(raw_audio))
                    s = s + volume_db 
                    out = io.BytesIO()
                    s.export(out, format="wav")
                    return out.getvalue()
                except Exception as vol_err:
                    print(f"Volume Adjust Error: {vol_err}")
                    return raw_audio
            else:
                return raw_audio

    except Exception as e:
        print(f"Generate API Error: {e}")
        
    return None

def delete_audio_range(original_wav_bytes, start_sec, end_sec):
    """
    오디오의 특정 구간(start_sec ~ end_sec)을 물리적으로 삭제하고 이어 붙입니다.
    예: 13초~15초 구간을 삭제하면, 15초에 있던 소리가 13초로 당겨집니다.
    """
    try:
        if not original_wav_bytes: return None
        
        # pydub로 로드
        original = AudioSegment.from_wav(io.BytesIO(original_wav_bytes))
        
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # 범위 유효성 검사
        if start_ms < 0: start_ms = 0
        if end_ms > len(original): end_ms = len(original)
        if start_ms >= end_ms: return None # 시작이 끝보다 뒤면 작업 불가
        
        # [삭제 로직] 앞부분 + 뒷부분 (중간을 건너뜀)
        # Crossfade(0.05초)를 살짝 주면 뚝 끊기는 느낌 없이 자연스럽게 붙습니다.
        part_front = original[:start_ms]
        part_back = original[end_ms:]
        
        # 단순히 더하기(+)도 가능하지만, 아주 짧은 크로스페이드를 권장합니다.
        final_audio = part_front.append(part_back, crossfade=20) 
        # crossfade가 싫으면: final_audio = part_front + part_back
        
        output_io = io.BytesIO()
        final_audio.export(output_io, format="wav")
        return output_io.getvalue()
        
    except Exception as e:
        print(f"Delete Logic Error: {e}")
        return None

def patch_audio_segment(original_wav_bytes, new_wav_bytes, start_sec, end_sec):
    try:
        # [방어 로직] 데이터가 None이거나 비어있으면 즉시 중단
        if not original_wav_bytes or len(original_wav_bytes) < 100:
            print("Patch Skip: 원본 데이터가 비어있습니다.")
            return None
        if not new_wav_bytes or len(new_wav_bytes) < 100:
            print("Patch Skip: 새로운 데이터가 비어있습니다.")
            return None

        # [핵심 방어] 시작 코드가 RIFF가 아니면(특히 00 00 00 00이면) 중단
        if not original_wav_bytes.startswith(b'RIFF'):
            print(f"Patch Skip: 원본 데이터 헤더 오류 (Start code: {original_wav_bytes[:4]})")
            return None
        if not new_wav_bytes.startswith(b'RIFF'):
            # 새 데이터가 Raw PCM일 수도 있으니 경고만 하고 넘어가거나, Raw 변환 시도
            # 여기서는 안전하게 WAV 헤더가 없으면 중단시킴 (add_silence_padding을 거쳤다면 헤더가 있어야 함)
            print(f"Patch Skip: 새 데이터 헤더 오류 (Start code: {new_wav_bytes[:4]})")
            return None

        # 1. pydub 로드
        original = AudioSegment.from_wav(io.BytesIO(original_wav_bytes))
        replacement = AudioSegment.from_wav(io.BytesIO(new_wav_bytes))
        
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        
        # 2. 편집 수행
        if start_ms < 0: start_ms = 0
        if end_ms > len(original): end_ms = len(original)
        
        part_a = original[:start_ms]
        part_c = original[end_ms:]
        
        fade_duration = 50 
        if len(replacement) < fade_duration * 2:
            final_audio = part_a + replacement + part_c
        else:
            final_audio = part_a.fade_out(fade_duration) + replacement.fade_in(fade_duration).fade_out(fade_duration) + part_c.fade_in(fade_duration)
        
        output_io = io.BytesIO()
        final_audio.export(output_io, format="wav")
        return output_io.getvalue()
        
    except Exception as e:
        print(f"Patch Process Error: {e}")
        return None

def get_transcription_with_timestamps(api_key, audio_path):
    """
    [최종_V3] 문장 단위 강제 병합(Merger) + 길이 제한(Clamping) 적용 버전
    1. AI가 쉼표(,)나 호흡 단위로 끊어서 주더라도, 마침표(., ?, !)가 나올 때까지 뒷문장과 합체합니다.
    2. 오디오 실제 길이보다 긴 시간이 나오지 않도록 강제 절삭합니다.
    """
    client = genai.Client(api_key=api_key)
    
    # 1. [실측] 오디오 길이 측정
    real_duration = 0.0
    try:
        with wave.open(audio_path, 'rb') as w:
            frames = w.getnframes()
            rate = w.getframerate()
            real_duration = frames / float(rate)
    except:
        real_duration = 600.0 # 측정 실패 시 fallback

    with open(audio_path, "rb") as f: audio_bytes = f.read()

    # 2. [주입] 프롬프트 설정
    prompt = f"""
    Listen to the audio and transcribe it.
    
    **CONTEXT:**
    - Total duration: {real_duration:.2f} seconds.
    
    **TASK:**
    - Transcribe the audio into a JSON list of segments.
    - Ideally, split by FULL SENTENCES (ending with . ? !).
    - If there is a long pause, you might split earlier, but try to keep sentences together.
    
    **FORMAT:**
    [
      {{"start": 0.0, "end": 5.2, "text": "Sentence one."}},
      {{"start": 5.2, "end": 10.5, "text": "Sentence two."}}
    ]
    Use double quotes. Time in seconds (float).
    """
    
    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]

    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro", 
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt)
                ])
            ],
            config=types.GenerateContentConfig(safety_settings=safety_settings)
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            return []

        raw_text = response.text
        
        # JSON 파싱
        start_idx = raw_text.find('[')
        end_idx = raw_text.rfind(']')
        if start_idx == -1 or end_idx == -1: return []
        clean_text = raw_text[start_idx : end_idx+1]

        # 시간 포맷 보정 (MM:SS -> SS)
        def time_replacer(match):
            time_str = match.group(0)
            if ':' in time_str:
                parts = time_str.split(':')
                if len(parts) == 2:
                    return str(int(parts[0]) * 60 + float(parts[1]))
                elif len(parts) == 3: 
                    return str(int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2]))
            return time_str 

        clean_text = re.sub(r'\d+:\d+(\.\d+)?', time_replacer, clean_text)
        
        # 1차 파싱 데이터
        raw_data = json.loads(clean_text)
        
        # ---------------------------------------------------------
        # [핵심 로직] 문장 병합기 (Sentence Merger)
        # 마침표로 끝나지 않는 조각들을 하나로 합칩니다.
        # ---------------------------------------------------------
        merged_data = []
        buffer = None

        for item in raw_data:
            # 시간 유효성 체크 (시작이 끝보다 뒤거나, 시작이 전체 길이 넘으면 패스)
            if item['start'] >= item['end']: continue
            if item['start'] >= real_duration: continue
            
            # 끝 시간 보정 (Clamping)
            if item['end'] > real_duration: item['end'] = real_duration
            
            # 버퍼가 비어있으면 새로 시작
            if buffer is None:
                buffer = item
            else:
                # 버퍼가 있으면 이어 붙이기
                # 텍스트는 공백 추가해서 연결, 끝 시간은 현재 아이템의 끝 시간으로 확장
                buffer['text'] = buffer['text'].strip() + " " + item['text'].strip()
                buffer['end'] = item['end']
            
            # 현재 버퍼의 텍스트가 문장 부호로 끝나는지 확인
            current_text = buffer['text'].strip()
            if current_text and current_text[-1] in ['.', '?', '!']:
                merged_data.append(buffer)
                buffer = None # 버퍼 비우기 (다음 문장 시작 준비)
        
        # 루프가 끝났는데 버퍼에 남은 게 있다면 (마지막 문장에 마침표가 없는 경우 등)
        if buffer is not None:
            merged_data.append(buffer)

        return merged_data
        
    except Exception as e:
        print(f"Parsing Error: {e}")
        return []

# ==========================================
# [3-0] 유틸리티: 오디오 텍스트 동기화
# ==========================================
# [교체] 오디오 플레이어 (JS 제어용)
def render_seekable_player(audio_path, seek_time=0.0):
    """
    오디오 플레이어를 렌더링하고, seek_time이 변경되면 해당 위치로 이동시킵니다.
    """
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
        
    # HTML/JS: 오디오 로드 시 특정 시간으로 점프
    html_code = f"""
    <audio id="player" controls style="width: 100%;">
        <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
    </audio>
    <script>
        var audio = document.getElementById('player');
        // 페이지 로드(Rerun) 시 파이썬에서 넘겨준 시간으로 이동
        audio.currentTime = {seek_time};
        // 자동 재생 (선택 사항, 원치 않으면 주석 처리)
        // audio.play(); 
    </script>
    """
    st.components.v1.html(html_code, height=60)

# ==========================================
# [3] 유틸리티: 검증 (STT & Diff)
# ==========================================
def normalize_text_strict(text):
    text = re.sub(r'[^가-힣a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def compare_texts_diff(original, transcribed):
    clean_orig = normalize_text_strict(original)
    clean_trans = normalize_text_strict(transcribed)
    matcher = difflib.SequenceMatcher(None, clean_orig, clean_trans)
    similarity = matcher.ratio() * 100
    
    diff_html = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        orig_fragment = clean_orig[i1:i2]
        trans_fragment = clean_trans[j1:j2]
        if tag == 'equal': diff_html.append(f"<span style='color:black; opacity:0.7;'>{orig_fragment}</span>")
        elif tag == 'replace': diff_html.append(f"<span style='color:red; text-decoration:line-through; background-color:#ffe6e6; font-weight:bold;'>{orig_fragment}</span> <span style='color:green; background-color:#e6ffe6; font-weight:bold;'>{trans_fragment}</span>")
        elif tag == 'delete': diff_html.append(f"<span style='color:red; text-decoration:line-through; background-color:#ffe6e6; font-weight:bold;'>{orig_fragment}</span>")
        elif tag == 'insert': diff_html.append(f"<span style='color:blue; background-color:#e6f3ff; font-weight:bold;'>{trans_fragment}</span>")
    return similarity, "".join(diff_html)

def verify_audio_content(api_key, audio_path):
    client = genai.Client(api_key=api_key)
    with open(audio_path, "rb") as f: audio_data = f.read()
    prompt = "Listen and transcribe EXACTLY as spoken."
    try:
        response = client.models.generate_content(
            model=VERIFY_MODEL_ID,
            contents=[types.Content(role="user", parts=[types.Part.from_bytes(data=audio_data, mime_type="audio/wav"), types.Part.from_text(text=prompt)])]
        )
        return response.text
    except Exception as e: return f"Error: {e}"

def get_correction_suggestion(api_key, original, transcribed):
    client = genai.Client(api_key=api_key)
    
    # 프롬프트: '오류 탐지' + '해결책(대본 수정안) 제시'
    prompt = f"""
    당신은 AI 오디오북 제작을 위한 **'대본 수선 전문가(Script Doctor)'**입니다.
    [원본 텍스트]와 [STT(녹음된 발음)]를 비교하여, **치명적인 내용 오류**를 찾고,
    다음 생성 시 이를 방지할 수 있는 **구체적인 대본 수정안**을 제시하세요.
    
    [원본 텍스트]:
    {original}
    
    [STT (녹음된 내용)]:
    {transcribed}
    
    **[분석 및 처방 가이드]**
    1. **누락(Omission) 발생 시:**
       - **원인:** 문장이 너무 길거나 호흡이 가빠서 AI가 급하게 넘어가며 건너뜀.
       - **💡 대본 수정법:** 쉼표(,)나 마침표(.)를 추가하여 문장을 물리적으로 끊어주거나, 줄바꿈을 하세요.
    2. **오독(Mutation) 발생 시:**
       - **원인:** 한자어, 영어, 또는 발음이 모호한 단어.
       - **💡 대본 수정법:** 소리 나는 대로 한글 발음을 적거나(예: 'resume' -> '레쥬메'), 한 글자씩 띄어쓰기(예: '확.실.히').
    3. **순서 뒤바뀜/문법 파괴 시:**
       - **원인:** 문장 구조가 복잡하여 AI가 문맥을 재해석해버림.
       - **💡 대본 수정법:** 긴 문장을 두 개의 짧은 단문으로 쪼개세요.
    
    **[출력 형식]**
    오류가 발견된 경우에만 아래 형식으로 출력하세요. (완벽하면 "✅ 수정 불필요" 출력)
    
    **1. [오류 유형]** (간단한 원인 설명)
       - 🔴 **문제 구간:** "(오류가 발생한 원본 문장 부분)"
       - 🟢 **수정 제안:** "(AI가 실수하지 않도록 기호나 철자를 변경한 텍스트)"
    
    **[작성 예시]**
    **1. [단어 누락]** 문장이 길어서 뒷부분 '갑자기'를 건너뜀
       - 🔴 **문제 구간:** "그는 문을 열고 들어와서 갑자기 소리쳤다"
       - 🟢 **수정 제안:** "그는 문을 열고 들어와서, 갑자기, 소리쳤다" (쉼표로 호흡 강제)
       
    **2. [치명적 오독]** '창조적'을 '참조적'으로 발음함
       - 🔴 **문제 구간:** "창조적인 활동"
       - 🟢 **수정 제안:** "창.조.적인 활동" (강조점 활용)
    """
    
    try:
        response = client.models.generate_content(model=VERIFY_MODEL_ID, contents=[prompt])
        return response.text
    except: 
        return "수정 제안 분석 실패"

def verify_errors_with_timestamp(api_key, audio_path, original_text):
    """
    [강력 검증] 오디오를 먼저 '받아쓰기' 시킨 후 원문과 대조하여,
    AI가 대충 넘어가는 것을 방지하고 정밀하게 오류를 찾아냅니다.
    """
    client = genai.Client(api_key=api_key)
    
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()
        
    # 프롬프트 전략 변경: "비교해라" (X) -> "받아쓰고 틀린 그림 찾아라" (O)
    prompt = f"""
    You are a paranoid Forensic Audio Analyst. 
    Your goal is to catch ANY discrepancy between the audio and the script.
    
    [Original Script]:
    {original_text}
    
    **INSTRUCTIONS (Follow Step-by-Step):**
    1. **Listen & Transcribe:** First, listen to the audio carefully and form a mental verbatim transcript of exactly what was said.
    2. **Compare:** Compare your mental transcript against the [Original Script] word by word.
    3. **Detect Errors:** Flag ANY difference, specifically:
       - **SKIPPED (Omission):** A word/sentence in the script is NOT in the audio.
       - **CHANGED (Mutation):** The script says "Father" but audio says "Brother".
       - **ADDED (Insertion):** Audio contains words not in the script.
       - **ORDER (Reordering):** Words are spoken in the wrong order.
    
    **STRICT RULES:**
    - DO NOT say "No errors found" if there is even a single word missing.
    - Be extremely strict. Do not forgive slight changes.
    - Provide the timestamp where the error STARTS.
    
    **Output Format (Markdown Table):**
    | Time (MM:SS) | Error Type | Script Text | Audio (What was heard) | Correction Suggestion |
    | :--- | :--- | :--- | :--- | :--- |
    | 00:05 | Omission | "He opened the door" | (Silence/Skipped) | Insert missing phrase |
    | 01:12 | Mutation | "Resume" | "Re-zoom" | Pronounce as "Re-zu-may" |
    
    If absolutely NO errors exist after checking twice, print: "✅ **Perfect Match: Verified.**"
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro", # 만약 여전히 못 찾으면 'gemini-1.5-pro'로 변경 권장
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text="[Original Script]:"),
                    types.Part.from_text(text=original_text),
                    types.Part.from_text(text="[Audio File]:"),
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt)
                ])
            ]
        )
        return response.text
    except Exception as e:
        return f"검증 실패: {e}"

def analyze_voice_similarity(api_key, ref_audio_bytes, target_audio_path):
    """
    [검증용] 기준 오디오(Reference)와 생성된 오디오(Generated)를 비교하여
    목소리 톤, 감정, 속도 등의 일치율을 분석함.
    """
    client = genai.Client(api_key=api_key)
    
    # 생성된 오디오 파일 읽기
    with open(target_audio_path, "rb") as f:
        target_bytes = f.read()
        
    prompt = """
    Listen to the two audio files provided.
    Audio 1 is the [Reference Voice] (Ground Truth).
    Audio 2 is the [Generated Voice] (Target).
    
    Compare Audio 2 against Audio 1 specifically on:
    1. Tone & Emotion (Mood)
    2. Speaking Speed (Pacing)
    3. Voice Identity Similarity (Does it sound like the same person/style?)
    
    Provide a 'Similarity Score' (0-100%) and a brief explanation.
    
    Format:
    **Similarity Score:** XX%
    **Analysis:** (Brief reason in 1-2 sentences)
    """
    
    try:
        response = client.models.generate_content(
            model=ANALYZE_MODEL_ID,  # gemini-2.0-flash-exp (멀티모달 지원)
            contents=[
                types.Content(role="user", parts=[
                    # [수정] text=... 를 명시하여 키워드 인자로 전달해야 함
                    types.Part.from_text(text="Audio 1 (Reference):"),
                    types.Part.from_bytes(data=ref_audio_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text="Audio 2 (Generated):"),
                    types.Part.from_bytes(data=target_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt)
                ])
            ]
        )
        return response.text
    except Exception as e:
        return f"분석 실패: {e} (API 키나 모델 권한을 확인하세요)"

def analyze_voice_style_to_prompt(api_key, ref_audio_bytes):
    """
    [2단계용] 오디오를 듣고 그 특징을 '연기 지시 텍스트'로 변환
    """
    client = genai.Client(api_key=api_key)
    
    prompt = """
    You are an expert Voice Engineer and Acting Coach.
    Listen to the attached audio file deeply.
    
    Analyze the voice waveform characteristics and convert them into specific acting instructions (System Prompt) for an AI TTS model.
    
    Focus on these specific details:
    1. **Pitch & Range**: (e.g., "Very Low Bass", "Mid-range Baritone", "High Soprano")
    2. **Timbre/Texture**: (e.g., "Raspy", "Breathiy", "Clear", "Gravelly")
    3. **Pacing & Rhythm**: (e.g., "Staccato (choppy)", "Legato (smooth)", "Fast-paced", "Slow and deliberate")
    4. **Intonation**: (e.g., "Flat/Monotone", "Dynamic/Expressive", "Rising at the end")
    
    **Output Instruction (in Korean):**
    Write a concise but highly descriptive instruction that forces the AI to mimic this exact sound. 
    Start with: "당신은 [성격/특징] 목소리를 가진 성우입니다..."
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-preview", # 분석은 멀티모달 모델 사용
            contents=[
                types.Content(role="user", parts=[
                    types.Part.from_text(text="Reference Audio:"),
                    types.Part.from_bytes(data=ref_audio_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text=prompt)
                ])
            ]
        )
        return response.text.strip()
    except Exception as e:
        return f"분석 실패: {e}"

# ==========================================
# [4] 메인 UI
# ==========================================
def main():
    st.set_page_config(layout="wide", page_title="AI AudioBook Workstation")
    
    st.sidebar.header("⚙️ 환경 설정")
    api_key = st.sidebar.text_input("Google API Key", type="password")
    if api_key: os.environ["GOOGLE_API_KEY"] = api_key
    
    st.title("🎙️ AI AudioBook : 3-Step Workstation")
    
    tab1, tab2, tab3 = st.tabs(["1️⃣ 텍스트 추출 & 대본화", "2️⃣ 오디오 자동 생성 (Batch)", "3️⃣ 품질 검증 & 수정"])

    # ----------------------------------------------------------------
    # TAB 1: 텍스트 분할 (UI 원복: 좌우 분할)
    # ----------------------------------------------------------------
    with tab1:
        st.subheader("1단계: EPUB 추출 및 대본 파일 생성")
        
        # [A] 파일 업로드 및 로드
        col_load1, col_load2 = st.columns(2)
        with col_load1:
            uploaded_file = st.file_uploader("EPUB 파일 업로드", type=["epub"])
        with col_load2:
            if os.path.exists(REMAINING_FILE):
                st.info("📂 이전에 작업하던 텍스트가 있습니다.")
                if st.button("이어하기 (저장된 텍스트 불러오기)"):
                    st.session_state['full_source'] = load_text_file(REMAINING_FILE)
                    st.rerun()

        if uploaded_file and st.button("텍스트 추출 시작"):
            text = get_full_text_from_epub(uploaded_file)
            st.session_state['full_source'] = text
            save_text_file(REMAINING_FILE, text)
            st.success("추출 완료!")
            st.rerun()

        # [B] 좌우 2단 분할 UI
        if 'full_source' in st.session_state:
            st.divider()
            
            # 좌: 원본 / 우: 챕터생성
            col_src, col_dst = st.columns([1, 1])
            
            # --- 좌측: 원본 텍스트 에디터 ---
            with col_src:
                st.markdown("### 📜 원본 텍스트 (Ctrl+X)")
                # key를 지정하여 수정 시 session_state에 자동 반영
                new_source = st.text_area(
                    "Source Text", 
                    value=st.session_state['full_source'], 
                    height=600, 
                    label_visibility="collapsed",
                    key="source_editor"
                )
                # 텍스트가 변경되면 변수에 반영 (Streamlit 동작 특성상 key 바인딩으로 자동 처리됨)

            # --- 우측: 챕터 생성 폼 ---
            with col_dst:
                st.markdown("### ✂️ 챕터 생성 (Ctrl+V)")
                
                with st.form("split_form"):
                    next_num = len(get_subdirectories(SCRIPT_ROOT)) + 1
                    
                    # [UI 추가] 1. 챕터 제목
                    chapter_title = st.text_input("챕터 제목", f"Chapter {next_num:02d}")
                    
                    # [UI 추가] 2. 분할 글자 수 설정 (슬라이더/숫자입력)
                    # 기본값을 500자로 설정 (오류 관리하기 딱 좋은 문단 단위 크기)
                    split_limit = st.number_input(
                        "✂️ 파일당 최대 글자 수 (권장: 300~600자)", 
                        min_value=100, 
                        max_value=3000, 
                        value=500, 
                        step=50,
                        help="이 글자 수를 넘어가면 문장(.) 단위로 잘라서 다음 파일로 넘깁니다.\n작을수록 수정 관리가 쉽고, 클수록 문맥 연결이 자연스럽습니다."
                    )

                    # [UI 추가] 3. 내용 입력
                    chapter_content = st.text_area("챕터 내용 (여기에 붙여넣기)", height=450)
                    
                    submit_btn = st.form_submit_button("💾 챕터 저장 및 변환")

                    if submit_btn:
                        if not chapter_content.strip():
                            st.error("내용이 비어있습니다.")
                        else:
                            # 1. 폴더 및 파일 생성
                            safe_title = sanitize_filename(f"{next_num:02d}_{chapter_title}")
                            path = os.path.join(SCRIPT_ROOT, safe_title)
                            os.makedirs(path, exist_ok=True)
                            
                            # 원본 저장
                            save_text_file(os.path.join(path, "raw.txt"), chapter_content)
                            
                            # 2. 전처리 및 분할 (사용자가 입력한 split_limit 적용)
                            # process_text_for_playground 함수에 max_chars 인자를 전달
                            chunks = process_text_for_playground(chapter_content, max_chars=split_limit)
                            
                            for i, chunk in enumerate(chunks):
                                base = f"script_{i+1:03d}"
                                save_text_file(os.path.join(path, f"{base}.txt"), chunk)
                                with open(os.path.join(path, f"{base}.json"), "w", encoding="utf-8") as f:
                                    json.dump({"segments": [{"text": chunk}]}, f, ensure_ascii=False, indent=2)
                            
                            # 3. 남은 원본 텍스트 업데이트 (좌측 에디터 내용 저장)
                            st.session_state['full_source'] = new_source
                            save_text_file(REMAINING_FILE, new_source)
                            
                            st.success(f"✅ '{safe_title}' 생성 완료! (설정: {split_limit}자 단위 → 총 {len(chunks)}개 파일)")
                            st.rerun()


    # ----------------------------------------------------------------
    # TAB 2: 오디오 생성 (안전한 복사-붙여넣기 방식 적용)
    # ----------------------------------------------------------------
    with tab2:
        st.subheader("2단계: AI 오디오 일괄 생성")
        
        # [A] 프롬프트 템플릿 정의
        PROMPT_TEMPLATES = {
            "직접 작성 (Direct Input)": "",
            
            "추천 1: 🎭 감성 오디오북 (소설/문학)": """당신은 베테랑 오디오북 성우입니다. 텍스트를 단순히 낭독하지 말고, 청자가 이야기 속 장면에 몰입할 수 있도록 '연기'해주세요.
[지침]
1. 대화문(" ")은 등장인물의 성격에 맞춰 목소리 톤을 살짝 바꿔주세요.
2. 지문(해설)은 차분하고 따뜻한 어조로 읽어주세요.
3. 쉼표나 마침표에서는 충분히 호흡을 두어, 여유를 주세요.""",
            
            "추천 2: 🎙️ 심야 라디오 DJ (에세이/편안함)": """당신은 심야 라디오 프로그램의 따뜻한 DJ입니다. 청자에게 1:1로 이야기를 들려주듯이 부드럽고 친근하게 말해주세요.
[지침]
- 문장을 끝맺을 때 딱딱하게 끊지 말고, 자연스럽게 여운을 남겨주세요.
- 발음은 정확해야 하지만, 아나운서처럼 딱딱하지 않게 '대화하듯' 자연스러워야 합니다.
- 전체적으로 차분하고 안정감 있는 '중저음'의 톤을 유지하세요.""",
            
            "추천 3: 🎬 영화 같은 연출 (판타지/몰입)": """이 텍스트는 영화의 한 장면입니다. 당신은 이 장면을 목소리로 연기하는 배우입니다. 상황 묘사와 대사 하나하나에 담긴 감정선(긴장감, 기쁨, 슬픔 등)을 생생하게 표현해주세요.
[지침]
- 단순히 글자를 읽는 것이 아니라, 텍스트 뒤에 숨겨진 감정을 읽어내야 합니다.
- 상황이 긴박하면 템포를 조금 빠르게, 차분한 상황이면 느리게 조절하여 리듬감을 만드세요."""
        }

        # [B] 템플릿 변경 콜백 (드롭다운 바꿀 때만 입력창 갱신)
        def on_template_change():
            selected = st.session_state["template_selector"]
            new_text = PROMPT_TEMPLATES[selected]
            if selected == "직접 작성 (Direct Input)" and not new_text:
                new_text = "차분하고 자연스럽게 읽어주세요."
            st.session_state["tab2_prompt"] = new_text

        # 폴더 로드 및 경로 설정
        folders = get_subdirectories(SCRIPT_ROOT)
        if not folders:
            st.warning("생성된 대본 폴더가 없습니다.")
        else:
            sel_folder = st.selectbox("작업할 챕터 폴더", folders)
            script_path = os.path.join(SCRIPT_ROOT, sel_folder)
            audio_path = os.path.join(AUDIO_ROOT, sel_folder)
            os.makedirs(audio_path, exist_ok=True)
            
            txt_files = get_files_in_dir(script_path, ".txt")
            txt_files = [f for f in txt_files if f != "raw.txt"]
            existing_audios = get_files_in_dir(audio_path, ".wav")
            start_idx = len(existing_audios)
            
            col_info, col_set = st.columns([1, 2])
            with col_info:
                st.info(f"📄 총 대본 파일: {len(txt_files)}개")
                st.warning(f"🚀 생성 대기중: {len(txt_files) - start_idx}개")
            
            with col_set:
                st.markdown("#### 🎛️ 생성 설정")
                c1, c2 = st.columns(2)
                
                with c1:
                    voice = st.selectbox("성우 선택", VOICE_OPTIONS, index=0)
                    
                    # [온도와 속도를 나란히 배치하거나 위아래로 배치]
                    col_opt1, col_opt2, col_opt3 = st.columns(3) # 컬럼을 3개로 늘림
                    with col_opt1:
                        temp = st.slider("감정 온도", 0.0, 2.0, 1.0, 0.1)
                    with col_opt2:
                        speed = st.slider("속도 (배속)", 0.5, 2.0, 1.0, 0.1)
                    with col_opt3:
                        # [추가됨] 볼륨 슬라이더
                        volume = st.slider("볼륨 (dB)", -10.0, 10.0, 0.0, 1.0, help="+3dB는 소리가 1.4배 커집니다.")
                    
                    # 템플릿 선택 (변경 시에만 콜백 실행)
                    st.selectbox(
                        "프롬프트 프리셋", 
                        list(PROMPT_TEMPLATES.keys()), 
                        index=1,
                        key="template_selector",
                        on_change=on_template_change
                    )
                
                with c2:
                    # 초기값 세팅
                    if "tab2_prompt" not in st.session_state:
                        st.session_state["tab2_prompt"] = PROMPT_TEMPLATES["추천 1: 🎭 감성 오디오북 (소설/문학)"]

                    # 메인 프롬프트 입력창
                    prompt_text = st.text_area("시스템 프롬프트 (연기 지시)", key="tab2_prompt", height=200)

                # ---------------------------------------------------------
                # [2단계 전용] 오디오 스타일 분석기 (별도 결과 표시)
                # ---------------------------------------------------------
                st.markdown("---")
                st.markdown("##### 🎤 목소리 스타일 추출 (따라하기)")
                
                col_a1, col_a2 = st.columns([2, 3])
                with col_a1:
                    style_audio = st.file_uploader("따라할 목소리 업로드 (분석용)", type=["wav", "mp3"], key="style_ref_tab2")
                
                with col_a2:
                    st.write("") 
                    st.write("")
                    
                    # 분석 결과 저장용 세션 변수
                    if "analysis_result_text" not in st.session_state:
                        st.session_state["analysis_result_text"] = ""

                    if style_audio:
                        if st.button("스타일 분석 실행 🔍"):
                            if not api_key: st.error("API Key 필요")
                            else:
                                with st.spinner("목소리 분석 중..."):
                                    extracted_style = analyze_voice_style_to_prompt(api_key, style_audio.getvalue())
                                    st.session_state["analysis_result_text"] = extracted_style
                                    st.success("분석 완료! 아래 텍스트를 복사해서 위 프롬프트에 추가하세요.")

                # 분석 결과가 있으면 표시 (사용자가 복사할 수 있게)
                if st.session_state["analysis_result_text"]:
                    st.info("👇 **분석된 스타일 지시문 (복사해서 위의 프롬프트 창에 붙여넣으세요)**")
                    st.code(st.session_state["analysis_result_text"], language="text")

                st.markdown("---")
                batch_count = st.number_input("이번에 생성할 개수", min_value=1, max_value=max(1, len(txt_files)-start_idx), value=min(5, max(1, len(txt_files)-start_idx)))

            if st.button("🎙️ 오디오 생성 시작 (이어하기)"):
                if not api_key:
                    st.error("API Key를 입력해주세요.")
                else:
                    # [NEW] 설정값 저장 (Metadata Saving)
                    # 나중에 3단계에서 불러올 수 있도록 설정값을 JSON으로 박제합니다.
                    settings_data = {
                        "voice": voice,
                        "speed": speed,
                        "temperature": temp,
                        "prompt": prompt_text
                    }
                    with open(os.path.join(audio_path, "settings.json"), "w", encoding="utf-8") as f:
                        json.dump(settings_data, f, ensure_ascii=False, indent=2)
                    
                    st.toast("⚙️ 생성 설정이 저장되었습니다.")

                    # --- 기존 생성 로직 ---
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    target_files = txt_files[start_idx : start_idx + batch_count]
                    
                    if not target_files:
                         st.info("생성할 대상 파일이 없습니다.")
                    else:
                        for i, txt_file in enumerate(target_files):
                            current_idx = start_idx + i + 1
                            status_text.markdown(f"**[{current_idx}/{len(txt_files)}] 생성 중...** `{txt_file}`")
                            text_content = load_text_file(os.path.join(script_path, txt_file))
                            
                            try:
                                audio_bytes = generate_speech(api_key, text_content, prompt_text, voice, temp, speed)
                                
                                if audio_bytes:
                                    final_audio = add_silence_padding(audio_bytes)
                                    if final_audio:
                                        save_name = txt_file.replace("script_", "audio_").replace(".txt", ".wav")
                                        with open(os.path.join(audio_path, save_name), "wb") as f:
                                            f.write(final_audio)
                                        status_text.markdown(f"✅ **성공:** `{save_name}`")
                                    else:
                                        st.error(f"패딩 오류: {txt_file}")
                                else:
                                    st.error(f"생성 실패: {txt_file}")
                                    break
                            except Exception as e:
                                st.error(f"에러: {e}")
                                break
                            
                            progress_bar.progress((i + 1) / batch_count)
                            time.sleep(1)
                        st.success("완료!")
                        time.sleep(2)
                        st.rerun()


    # ----------------------------------------------------------------
    # TAB 3: 검증 및 수정 (클릭 연동 업그레이드 버전)
    # ----------------------------------------------------------------
    with tab3:
        st.subheader("3단계: 품질 검증 및 정밀 수선")
        
        folders = get_subdirectories(SCRIPT_ROOT)
        if not folders:
            st.warning("작업할 폴더가 없습니다.")
        else:
            sel_folder = st.selectbox("검증할 폴더", folders, key="verify_folder")
            script_path = os.path.join(SCRIPT_ROOT, sel_folder)
            audio_path = os.path.join(AUDIO_ROOT, sel_folder)
            
            # 설정 파일 로드
            loaded_voice_idx = 0
            loaded_speed = 1.0
            loaded_prompt = "자연스럽게 연결되도록 읽어주세요."
            
            settings_file = os.path.join(audio_path, "settings.json")
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, "r", encoding="utf-8") as f:
                        saved_settings = json.load(f)
                    if saved_settings.get("voice") in VOICE_OPTIONS:
                        loaded_voice_idx = VOICE_OPTIONS.index(saved_settings.get("voice"))
                    loaded_speed = float(saved_settings.get("speed", 1.0))
                    loaded_prompt = saved_settings.get("prompt", "")
                except: pass

            json_files = get_files_in_dir(script_path, ".json")
            if json_files:
                target_script = st.selectbox("대본 파일 선택", json_files)
                target_audio = target_script.replace("script_", "audio_").replace(".json", ".wav")
                audio_full_path = os.path.join(audio_path, target_audio)

                # 세션 상태 초기화
                if "input_start" not in st.session_state: st.session_state["input_start"] = 0.0
                if "input_end" not in st.session_state: st.session_state["input_end"] = 0.0
                if "input_text" not in st.session_state: st.session_state["input_text"] = ""
                
                # 타임라인 데이터 확보
                verify_key = f"timeline_{target_script}"
                if verify_key not in st.session_state:
                    if st.button("🚀 분석 데이터 생성 (최초 1회 필수)"):
                        if not api_key: st.error("API Key 필요")
                        else:
                            with st.spinner("오디오 분석 중..."):
                                timeline_data = get_transcription_with_timestamps(api_key, audio_full_path)
                                st.session_state[verify_key] = timeline_data
                                st.rerun()

                st.divider()

                # [화면 분할] 좌: 리스트(선택) / 우: 에디터
                col_list, col_edit = st.columns([1.2, 1])
                
                # ==========================================
                # [좌측] 인터랙티브 리스트 (클릭 -> 입력 연동)
                # ==========================================
                with col_list:
                    st.markdown("### 📜 문장 선택 (클릭)")
                    
                    if os.path.exists(audio_full_path):
                        # 1. 오디오 플레이어 (현재 선택된 시간으로 렌더링)
                        render_seekable_player(audio_full_path, seek_time=st.session_state["input_start"])
                        
                        # 2. 문장 리스트 (버튼)
                        if verify_key in st.session_state and st.session_state[verify_key]:
                            timeline = st.session_state[verify_key]
                            
                            # 스크롤 가능한 컨테이너 (높이 고정)
                            with st.container(height=500):
                                for i, seg in enumerate(timeline):
                                    # 현재 선택된 문장이면 색상 강조 (primary)
                                    is_selected = (seg['start'] == st.session_state["input_start"])
                                    btn_type = "primary" if is_selected else "secondary"
                                    
                                    # 버튼 라벨: [시간] 텍스트
                                    label = f"[{seg['start']:.1f}s ~ {seg['end']:.1f}s] {seg['text']}"
                                    
                                    # 버튼 클릭 시 -> 세션 상태 업데이트 -> Rerun
                                    if st.button(label, key=f"btn_{i}", type=btn_type, use_container_width=True):
                                        st.session_state["input_start"] = seg['start']
                                        st.session_state["input_end"] = seg['end']
                                        st.session_state["input_text"] = seg['text']
                                        st.rerun()
                        else:
                            st.info("👆 위 '분석 데이터 생성' 버튼을 눌러주세요.")
                    else:
                        st.error("오디오 파일이 없습니다.")

                # ==========================================
                # [우측] 오디오 편집 (자동 싱크 보정 적용됨)
                # ==========================================
                with col_edit:
                    st.markdown("### ✂️ 1. 여백/구간 삭제")
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        del_start = st.number_input("삭제 시작", min_value=0.0, step=0.1, format="%.2f", key="del_start")
                    with dc2:
                        del_end = st.number_input("삭제 끝", min_value=0.0, step=0.1, format="%.2f", key="del_end")
                    
                    if st.button("🗑️ 구간 삭제 실행", type="primary", use_container_width=True):
                         if os.path.exists(audio_full_path):
                            with open(audio_full_path, "rb") as f: original_bytes = f.read()
                            
                            new_audio_bytes = delete_audio_range(original_bytes, del_start, del_end)
                            
                            if new_audio_bytes:
                                # 1. 오디오 저장
                                with open(audio_full_path, "wb") as f: f.write(new_audio_bytes)
                                
                                # 2. [핵심] 타임라인 데이터 즉시 보정 (API 호출 X)
                                if verify_key in st.session_state:
                                    old_timeline = st.session_state[verify_key]
                                    # 수학적 계산으로 시간 당기기
                                    new_timeline = adjust_timeline_for_deletion(old_timeline, del_start, del_end)
                                    st.session_state[verify_key] = new_timeline
                                    st.toast(f"✅ 오디오 삭제 완료! 타임라인도 {del_end-del_start:.1f}초 당겨졌습니다.")
                                else:
                                    st.success("삭제 완료")
                                
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.error("삭제 실패")

                    st.divider()

                    st.markdown("### 🛠️ 2. 문장 재녹음")
                    tc1, tc2 = st.columns(2)
                    with tc1:
                        start_time = st.number_input("시작(초)", step=0.1, format="%.2f", key="input_start")
                    with tc2:
                        end_time = st.number_input("끝(초)", step=0.1, format="%.2f", key="input_end")
                    
                    patch_text = st.text_area("수정할 내용", height=100, key="input_text")

                    with st.expander("⚙️ 성우 / 속도 / 볼륨 설정", expanded=False):
                        pc1, pc2, pc3 = st.columns(3)
                        with pc1:
                            patch_voice = st.selectbox("성우", VOICE_OPTIONS, index=loaded_voice_idx, key="patch_voice")
                        with pc2:
                            patch_speed = st.slider("속도", 0.5, 2.0, loaded_speed, 0.1, key="patch_speed")
                        with pc3:
                            # [추가됨] 재녹음용 볼륨
                            patch_vol = st.slider("볼륨(dB)", -10.0, 10.0, 0.0, 1.0, key="patch_vol")
                            
                        patch_prompt = st.text_input("스타일 지시", loaded_prompt, key="patch_prompt")

                    if st.button("🩹 재녹음 및 덮어씌우기", type="secondary", use_container_width=True):
                        if not patch_text.strip():
                            st.error("내용이 없습니다.")
                        else:
                            with st.spinner("생성 중..."):
                                # 1. 오디오 생성
                                raw_audio = generate_speech(api_key, patch_text, patch_prompt, patch_voice, 1.0, patch_speed, patch_vol)
                                new_part = add_silence_padding(raw_audio, duration_sec=0.1)
                                
                                if new_part and os.path.exists(audio_full_path):
                                    # 생성된 오디오 길이 측정 (중요: 정확한 보정을 위해)
                                    import wave
                                    with wave.open(io.BytesIO(new_part), 'rb') as w:
                                        frames = w.getnframes()
                                        rate = w.getframerate()
                                        new_duration = frames / float(rate)

                                    with open(audio_full_path, "rb") as f: orig = f.read()
                                    patched = patch_audio_segment(orig, new_part, start_time, end_time)
                                    
                                    if patched:
                                        # 2. 오디오 저장
                                        with open(audio_full_path, "wb") as f: f.write(patched)
                                        
                                        # 3. [핵심] 타임라인 데이터 즉시 보정
                                        if verify_key in st.session_state:
                                            old_timeline = st.session_state[verify_key]
                                            # 수학적 계산으로 시간 밀기/당기기
                                            new_timeline = adjust_timeline_for_patch(old_timeline, start_time, end_time, new_duration)
                                            st.session_state[verify_key] = new_timeline
                                            st.toast("✅ 재녹음 완료! 타임라인 싱크가 자동 조절되었습니다.")
                                        else:
                                            st.success("수선 완료")

                                        time.sleep(0.5)
                                        st.rerun()

if __name__ == "__main__":
    main()