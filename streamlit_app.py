import streamlit as st
import pandas as pd
import jieba
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image

# --- 中文字型設定 ---
font_path = "msyh.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# --- 載入與前處理資料 ---
df = pd.read_csv("104_jobs_all.csv")
df = df.dropna(subset=['jobName', 'description', 'jobAddrNoDesc'])

def extract_content_and_qualification(text):
    pattern_content = re.compile(r'(?:內容|職責|do(?:[】•])*)(.*?)(?=(?:資格|待遇|報名|期間|條件|$))', re.DOTALL)
    match_content = pattern_content.search(text)
    jobContent = match_content.group(1).strip() if match_content else text.strip()

    pattern_qualification = re.compile(r'(?:條件|資格)[：:](.*?)(?=(?:工作地點|待遇|條件|要求|$))', re.DOTALL)
    match_qualification = pattern_qualification.search(text)
    jobQualification = match_qualification.group(1).strip() if match_qualification else ""

    return pd.Series([jobContent, jobQualification], index=["jobContent", "jobQualification"])

def extract_city(text):
    pattern_city = re.compile(r'^([^市縣]+)(?=[市縣])', re.DOTALL)
    match_city = pattern_city.search(text)
    return match_city.group(1).strip() if match_city else ""

df_content = df['description'].apply(extract_content_and_qualification)
df_city = df['jobAddrNoDesc'].apply(extract_city)
df_new = pd.concat([df[['jobName']], df_content, df_city.rename("jobCity")], axis=1)

# --- 中文斷詞與詞頻分析 ---
text_data = " ".join(df_new['jobContent'].dropna().tolist())
jieba.setLogLevel(20)
stopwords = set(["我們", "and", "相關", "公司", "提供", "\r\n", "熟悉", "升遷", "00", "to", "的", "是", "在", "有", "為", "和", "為了", 
                 "那", "有些", "中", "這", "此", "上", "下", "與", "有關", "必須", "會", "之", "而", "應", "一", "也", "各", "所", "即",
                 "等", "等於", "或者", "做", "可以", "應該", "會", "想", "來", "去", "嗎", "是的", "對", "沒有", "the"])
words = [w for w in jieba.cut(text_data) if len(w) > 1 and w not in stopwords]
word_counts = Counter(words)

# --- TF-IDF & LDA ---
def jieba_tokenizer(text):
    return [word for word in jieba.cut(text) if len(word) > 1 and word not in stopwords]

documents = df_new['jobContent'].dropna().tolist()
vectorizer_lda = CountVectorizer(tokenizer=jieba_tokenizer)
X = vectorizer_lda.fit_transform(documents)
lda = LatentDirichletAllocation(n_components=3, max_iter=30, random_state=0)
lda.fit(X)

# --- Streamlit UI 設定 ---
st.set_page_config(page_title="找實習 Chatbot", page_icon="💬")
st.title("💬 Job Chatbot 實習助理")

# --- 圖片上傳功能 ---
st.markdown("### 🖼️ 上傳一張圖片")
uploaded_file = st.file_uploader("選擇一張圖片", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="你上傳的圖片", use_column_width=True)

# --- 對話提示 ---
st.markdown("""
歡迎使用找實習 Chatbot！

**你可以這樣問我：**
- 找資料分析相關的實習
- 顯示熱門詞彙
- 哪個城市實習最多？
- 顯示主題分析結果
- 畫出城市職缺柱狀圖
""")

# --- 儲存聊天訊息狀態 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- 回應生成邏輯 ---
def generate_response(prompt):
    prompt = prompt.lower()
    keyword_match = re.search(r'找(.*?)(?:相關)?的?(實習)?', prompt)
    if keyword_match:
        keyword = keyword_match.group(1).strip()
        is_intern = keyword_match.group(2) is not None
        return find_job_by_keyword(keyword, is_intern)
    elif "熱門" in prompt or "詞彙" in prompt:
        return get_top_keywords()
    elif "主題" in prompt:
        return show_topics()
    elif "畫出" in prompt and "城市" in prompt:
        show_top_city_bar()
        return "此圖是前 10 城市的職缺分布圖表。"
    elif "哪個城市" in prompt or "最多職缺" in prompt:
        return city_distribution()
    else:
        return "我可以幫你查詢實習職缺、熱門詞彙、主題或城市分布，請試著輸入「找資料分析相關的實習」這類問題"

def find_job_by_keyword(keyword, is_intern=False):
    # 搜尋職務內容欄位中包含關鍵字的職缺
    df_filtered = df_new[
        df_new['jobContent'].str.contains(keyword, na=False, case=False)
    ]
    if is_intern:
        df_filtered = df_filtered[
            df_filtered['jobName'].str.contains("實習", na=False) |
            df_filtered['jobContent'].str.contains("實習", na=False)
        ]
    jobs = df_filtered['jobName'].head(5).tolist()
    if jobs:
        return f"與「{keyword}」相關的前 5 筆{'實習' if is_intern else '工作'}：\n" + "\n".join(jobs)
    else:
        return f"找不到與「{keyword}」相關的職缺，請試試其他關鍵字。"

def get_top_keywords():
    top_words = word_counts.most_common(10)
    return "熱門詞彙：\n" + "\n".join([f"{w}（{c}次）" for w, c in top_words])

def show_topics():
    result = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [vectorizer_lda.get_feature_names_out()[i] for i in topic.argsort()[:-6:-1]]
        result.append(f"主題 {topic_idx + 1}：{'、'.join(top_words)}")
    return "\n".join(result)

def city_distribution():
    city_counts = df_new['jobCity'].value_counts().head(5)
    return "最多職缺的城市：\n" + "\n".join([f"{city}：{count}筆" for city, count in city_counts.items()])

def show_top_city_bar():
    city_counts = df_new['jobCity'].value_counts().head(10)
    fig, ax = plt.subplots()
    ax.bar(city_counts.index, city_counts.values, color="#6FA8DC")
    ax.set_title("前 10 城市職缺數量", fontproperties=font_prop)
    ax.set_xlabel("城市", fontproperties=font_prop)
    ax.set_ylabel("職缺數", fontproperties=font_prop)
    plt.xticks(rotation=45, fontproperties=font_prop)
    plt.yticks(fontproperties=font_prop)
    st.pyplot(fig)

# --- 處理使用者輸入 ---
if prompt := st.chat_input("請問你想知道什麼職缺資訊？", key="chat_bot"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = generate_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
