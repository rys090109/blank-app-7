# import streamlit as st

# st.title("🎈 My new app")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )
# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import io
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# -----------------------------
# ✅ 안정적 네트워크 (requests + Retry)
# -----------------------------
import socket
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_retries = Retry(
    total=3,
    backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)
_session.mount("https://", HTTPAdapter(max_retries=_retries))
_session.mount("http://", HTTPAdapter(max_retries=_retries))

# 배포 환경에서 IPv6 문제 회피용 (필요 없으면 False)
FORCE_IPV4 = True
if FORCE_IPV4:
    _orig_getaddrinfo = socket.getaddrinfo
    def _ipv4_only_getaddrinfo(host, port, *args, **kwargs):
        res = _orig_getaddrinfo(host, port, *args, **kwargs)
        v4 = [ai for ai in res if ai[0] == socket.AF_INET]
        return v4 or res
    socket.getaddrinfo = _ipv4_only_getaddrinfo

# -----------------------------
# ✅ 한국어 폰트 등록
# -----------------------------
from matplotlib import font_manager as fm, rcParams
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()
if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()
else:
    font_prop = fm.FontProperties()
rcParams["axes.unicode_minus"] = False

# -----------------------------
# Streamlit 설정
# -----------------------------
st.set_page_config(layout="wide", page_title="CO₂ & Global Temperature Dashboard")
st.title("🌍 대기 중 CO₂ 농도와 지구 평균 기온, 무슨 관계가 있을까?")
st.caption("데이터 출처: NOAA GML, NASA GISTEMP · 실패 시 data/co2_temp_merged_1960_2024.csv 사용")

# -----------------------------
# 안전한 fetch 유틸 (requests 기반)
# -----------------------------
def fetch_text(url: str, timeout: int = 12) -> list[str]:
    """urllib 대체. 세션/재시도/백오프/IPv4 우선."""
    headers = {"User-Agent": "Mozilla/5.0 (Streamlit classroom app)"}
    resp = _session.get(url, headers=headers, timeout=(6, timeout))
    resp.raise_for_status()
    return resp.text.replace("\r\n", "\n").splitlines()

def safe_fetch_lines(url: str, *, fallback_path: Path | None = None, timeout: int = 12) -> list[str] | None:
    """성공 시 텍스트 라인 반환, 실패 시 fallback_path가 존재하면 None(=로컬 사용 신호) 반환."""
    try:
        return fetch_text(url, timeout=timeout)
    except Exception:
        if fallback_path and fallback_path.exists():
            st.warning(f"⚠️ 원격 데이터 호출 실패 → 로컬 CSV 사용 ({fallback_path})")
            return None
        raise

# -----------------------------
# 데이터 로더 (원격 → 실패 시 data CSV)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_datasets() -> pd.DataFrame:
    co2_url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    temp_url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    fallback = Path("data/co2_temp_merged_1960_2024.csv")

    # 1) CO₂
    lines = safe_fetch_lines(co2_url, fallback_path=fallback)
    if lines is None:
        # 완성 병합본 CSV로 대체
        return pd.read_csv(fallback)

    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            val = float(parts[3])  # average column
        except Exception:
            continue
        rows.append([year, month, val])
    co2_df = pd.DataFrame(rows, columns=["year", "month", "co2_ppm"])
    co2_df = co2_df.groupby("year", as_index=False)["co2_ppm"].mean().rename(columns={"year": "Year"})

    # 2) GISTEMP
    lines = safe_fetch_lines(temp_url, fallback_path=fallback)
    if lines is None:
        return pd.read_csv(fallback)

    header_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("Year"))
    temp_df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    target_col = "J-D" if "J-D" in temp_df.columns else temp_df.columns[1]
    temp_df = temp_df[["Year", target_col]].rename(columns={target_col: "TempAnomaly"})
    temp_df["TempAnomaly"] = pd.to_numeric(temp_df["TempAnomaly"], errors="coerce")
    # 센티-섭씨 스케일이면 ℃로 변경
    if temp_df["TempAnomaly"].abs().median() > 5:
        temp_df["TempAnomaly"] = temp_df["TempAnomaly"] / 100.0
    temp_df = temp_df.dropna()

    merged = pd.merge(co2_df, temp_df, on="Year", how="inner").sort_values("Year").reset_index(drop=True)
    return merged

# -----------------------------
# 데이터 로드
# -----------------------------
with st.spinner("데이터 로딩 중... 잠시만 기다려 주세요! 🚀"):
    try:
        df = load_datasets()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 문제가 발생했습니다: {e}")
        st.stop()

# df 컬럼 가드
required_cols = {"Year", "co2_ppm", "TempAnomaly"}
if not required_cols.issubset(df.columns):
    st.error(f"필수 컬럼 누락: {required_cols - set(df.columns)}")
    st.stop()

# -----------------------------
# UI: 기간 선택
# -----------------------------
yr_min = int(df["Year"].min())
yr_max = int(df["Year"].max())

st.sidebar.header("연도 범위 선택")
yr_start, yr_end = st.sidebar.slider(
    "보고 싶은 기간을 골라보세요!",
    min_value=yr_min, max_value=yr_max,
    value=(max(1960, yr_min), yr_max), step=1
)
smooth = st.sidebar.checkbox("12년 이동평균 (전체적인 흐름 보기)", value=True)

df_r = df[(df["Year"] >= yr_start) & (df["Year"] <= yr_end)].copy()
if df_r.empty or len(df_r) < 2:
    st.warning("선택한 연도 범위에 데이터가 부족합니다. 범위를 넓혀 보세요.")
    st.stop()

if smooth and len(df_r) >= 12:
    df_r["co2_ppm_smooth"] = df_r["co2_ppm"].rolling(12, center=True, min_periods=1).mean()
    df_r["TempAnomaly_smooth"] = df_r["TempAnomaly"].rolling(12, center=True, min_periods=1).mean()

st.caption(
    f"적용 연도 범위: {int(df_r['Year'].min())}–{int(df_r['Year'].max())} "
    f"(전체 데이터 최신 연도: {int(df['Year'].max())})"
)

# -----------------------------
# 시각화
# -----------------------------
st.subheader("📈 CO₂ 농도와 지구 평균 기온, 같이 볼까요?")
sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10.5, 5.2))

# CO₂ (좌축)
ax1.plot(df_r["Year"], df_r["co2_ppm"], lw=1.6, color="#1f77b4", alpha=0.45, label="CO₂ 농도 (연평균)")
if smooth and "co2_ppm_smooth" in df_r.columns:
    ax1.plot(df_r["Year"], df_r["co2_ppm_smooth"], lw=2.8, color="#1f77b4", label="CO₂ 농도 (장기 추세)")
ax1.set_xlabel("연도", fontproperties=font_prop)
ax1.set_ylabel("대기 중 CO₂ (ppm)", color="#1f77b4", fontproperties=font_prop)
ax1.tick_params(axis="y", labelcolor="#1f77b4")

# 기온 이상치 (우축)
ax2 = ax1.twinx()
ax2.plot(df_r["Year"], df_r["TempAnomaly"], lw=1.6, color="#d62728", alpha=0.45, label="기온 변화 (연평균)")
if smooth and "TempAnomaly_smooth" in df_r.columns:
    ax2.plot(df_r["Year"], df_r["TempAnomaly_smooth"], lw=2.8, color="#d62728", label="기온 변화 (장기 추세)")
ax2.set_ylabel("지구 평균 기온 변화 (℃)", color="#d62728", fontproperties=font_prop)
ax2.tick_params(axis="y", labelcolor="#d62728")

plt.title(f"CO₂ 농도와 지구 평균 기온 변화 ({yr_start}–{yr_end})", pad=10, fontproperties=font_prop)

# 범례 통합
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, prop=font_prop)

fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# -----------------------------
# 요약 지표
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("CO₂ 얼마나 늘었을까?", f"{df_r['co2_ppm'].iloc[-1] - df_r['co2_ppm'].iloc[0]:+.1f} ppm")
c2.metric("기온은 얼마나 변했을까?", f"{df_r['TempAnomaly'].iloc[-1] - df_r['TempAnomaly'].iloc[0]:+.2f} ℃")
c3.metric("얼마나 관련 있을까? (상관계수)", f"{np.corrcoef(df_r['co2_ppm'], df_r['TempAnomaly'])[0,1]:.2f}")

with st.expander("데이터 표로 확인하기"):
    st.dataframe(
        df_r[["Year", "co2_ppm", "TempAnomaly"]]
          .rename(columns={"Year": "연도", "co2_ppm": "CO₂(ppm)", "TempAnomaly": "기온 변화(℃)"}),
        use_container_width=True
    )

# 병합 데이터 다운로드 (현재 구간)
csv_bytes = df_r.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 분석용 CSV 내려받기 (현재 구간 병합본)",
    data=csv_bytes,
    file_name=f"co2_temp_merged_{yr_start}_{yr_end}.csv",
    mime="text/csv"
)

# -----------------------------
# 📘 데이터 해석 (모둠 관점)
# -----------------------------
st.markdown("---")
st.header("📘 데이터 탐구 보고서: 우리 모둠의 발견")

st.subheader("1. 대기 중 CO₂ 농도의 지속적인 증가")
st.markdown("""
그래프의 파란색 선(CO₂ 농도)을 보면 알 수 있듯이, CO₂ 농도가 지속적으로 상승하는 모습은 저희 모둠에게 상당히 인상적이었습니다. 
저희가 태어나기 전인 1960년대 약 320ppm에서 현재 420ppm을 초과하는 수치에 도달한 것을 확인했습니다. 
이는 단순히 숫자의 변화를 넘어, 인류의 활동이 지구 대기 환경 전체에 영향을 미치고 있다는 명확한 증거라고 생각되어 책임감을 느끼게 되었습니다.
""")

st.subheader("2. '기온 이상치' 상승의 의미")
st.markdown("""
빨간색 선(기온 변화)으로 표시된 '기온 이상치'는 특정 기준값과의 차이를 의미합니다. NASA에서는 **1951년부터 1980년까지의 30년 평균 기온**을 그 기준으로 사용합니다. 
즉, 그래프의 0℃ 선이 바로 이 기간의 평균 기온이며, 각 연도의 값은 이 기준보다 얼마나 기온이 높았는지(플러스 값) 또는 낮았는지(마이너스 값)를 보여주는 것입니다.

분석 결과, 최근에는 기준치보다 매년 0.5℃ 이상 높았으며, 근래에는 1℃를 초과하는 해도 관측되었습니다. 
1℃라는 수치가 작게 느껴질 수 있지만, 이것이 전 지구적인 폭염, 폭우 등 극단적 기상 현상의 원인이 된다는 사실을 배우며 문제의 심각성을 체감할 수 있었습니다.
""")

st.subheader("3. CO₂ 농도와 기온 변화의 뚜렷한 상관관계")
st.markdown("""
이번 탐구에서 가장 주목할 만한 점은 **파란색 CO₂ 농도 선과 빨간색 기온 변화 선이** 매우 유사한 형태로 함께 상승한다는 사실이었습니다. 
CO₂ 농도가 증가함에 따라 기온 역시 상승하는 뚜렷한 경향성을 발견했습니다. 
이는 과학 시간에 배운 온실효과를 데이터로 직접 확인하는 과정이었으며, 눈에 보이지 않는 기체가 지구 전체의 온도를 높여 우리 삶에 직접적인 영향을 미칠 수 있다는 사실을 실감하게 했습니다.
""")

st.subheader("4. 탐구를 통해 느낀 점")
st.markdown("""
이번 프로젝트는 단순한 과제를 넘어, 데이터를 통해 미래 사회의 문제를 읽어내는 의미 있는 경험이었습니다. 
기후 위기가 막연한 미래의 일이 아닌, 우리가 살고 있는 현재의 문제임을 데이터를 통해 명확히 인식하게 되었습니다. 
이에 저희 모둠은 앞으로 교실 소등, 분리배출과 같은 일상 속 작은 실천부터 책임감을 갖고 행동하기로 다짐했습니다.
""")

# -----------------------------
# 📢 우리 세대를 위한 제언
# -----------------------------
st.markdown("---")
st.header("📢 우리 세대를 위한 제언")

st.markdown("""
저희는 이번 프로젝트를 통해 기후 위기가 교과서 속 지식이 아닌, 우리 모두의 현실임을 확인했습니다. 
따라서 같은 시대를 살아가는 학생들에게 다음과 같이 제안하고자 합니다.
""")

st.markdown("""
**1. 작은 실천의 중요성** 일상 속에서 무심코 사용하는 에너지를 절약하고, 급식 잔반을 남기지 않고, 일회용품 사용을 줄이는 등의 작은 습관이 모여 큰 변화를 만들 수 있습니다.

**2. 데이터 기반의 소통** "지구가 아프다"는 감성적인 호소와 더불어, 객관적인 데이터를 근거로 토론하고 소통할 때 더 큰 설득력을 가질 수 있습니다.

**3. 학교 공동체 내에서의 활동** 환경 동아리 활동이나 학급 캠페인을 통해 기후 위기 문제에 대한 공감대를 형성하고, 학교 차원의 해결 방안을 함께 고민해 볼 수 있습니다.

**4. 미래 진로와의 연계** 기후 위기 문제를 해결하기 위한 과학 기술, 사회 정책 등 관련 분야로의 진로를 탐색하는 것은 우리 세대가 미래를 준비하는 또 다른 방법이 될 것입니다.

기후 위기는 거대하고 어려운 문제이지만, 데이터를 통해 현상을 정확히 이해하고 함께 행동한다면 충분히 해결해 나갈 수 있습니다. 
우리 세대의 관심과 실천이 지속 가능한 미래를 만드는 첫걸음이 될 것이라고 믿습니다. 🌱
""")

# -----------------------------
# 📚 참고자료
# -----------------------------
st.markdown("---")
st.header("📚 참고자료")

st.markdown("""
- **데이터 출처**
    - [NOAA Global Monitoring Laboratory - Mauna Loa CO₂ Data](https://gml.noaa.gov/ccgg/trends/data.html)
    - [NASA GISS Surface Temperature Analysis (GISTEMP v4)](https://data.giss.nasa.gov/gistemp/)
- **추천 도서**
    - 그레타 툰베리, 《기후 책》, 이순희 역, 기후변화행동연구소 감수, 열린책들, 2023. 
      ([Yes24 도서 정보 링크](https://www.yes24.com/product/goods/119700330))
""")

# -----------------------------
# Footer (팀명)
# -----------------------------
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: gray; font-size: 0.9em;'>
        미림마이스터고 1학년 4반 2조 · 지구야아프지말아조
    </div>
    """,
    unsafe_allow_html=True
)
# streamlit_app.py
"""
Streamlit 대시보드 (한국어 UI)
- 주요 공개 데이터: World Bank CO2 emissions (per capita) 및 NASA GISTEMP (대체/참고)
  출처:
    - World Bank Indicator (CO2 per capita): https://api.worldbank.org/v2/indicator/EN.ATM.CO2E.PC
      (예시 API 사용 문서/튜토리얼: https://worldbank.github.io/template/notebooks/world-bank-api.html)
      (World Bank indicator page / docs: https://data.worldbank.org/)
    - NASA GISTEMP (global surface temperature): https://data.giss.nasa.gov/gistemp/
  (위 URL은 코드 주석으로 남겨두었음)
- 동작:
  1) 공개 데이터 대시보드 (자동으로 World Bank API에서 불러오기, 실패하면 예시 데이터로 대체)
  2) 사용자 입력 대시보드 (프롬프트의 Input 섹션에서 제공된 CSV/이미지/설명만 사용.
     현재 입력이 없으면 내부 예시 데이터로 대체하여 별도 대시보드 표시 — 앱 실행 중 업로드/입력 요구하지 않음)
- 규칙:
  - 날짜(미래) 데이터 제거 (로컬 기준 현재 시각의 연도/월 넘어가는 데이터 제거)
  - @st.cache_data 사용 (캐싱)
  - 전처리된 표 CSV 다운로드 버튼 제공
  - 모든 라벨/버튼/툴팁은 한국어
  - 폰트 시도: /fonts/Pretendard-Bold.ttf (없으면 무시)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import time
import datetime
import plotly.express as px
import wbgapi as wb

# ----------------------------
# 설정: (사용자 입력 섹션)
# ----------------------------
# 주의: 프롬프트의 Input 섹션에서 제공된 CSV/이미지/설명이 있다면
# 아래 USER_CSV_CONTENT 변수에 CSV 텍스트를 그대로 붙여넣어 주세요.
# (현재 대화에서는 Input 내용이 비어 있으므로 기본값 None으로 둡니다.)
USER_CSV_CONTENT = None  # <-- 사용자가 제공한 CSV 텍스트를 여기에 직접 넣지 않는 이상 None

# ----------------------------
# 유틸리티 및 상수
# ----------------------------
WORLD_BANK_INDICATOR = "EN.ATM.CO2E.PC"  # CO2 emissions (metric tons per capita)
WORLD_BANK_API_TEMPLATE = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&date={start}:{end}&per_page=20000"

# 폰트 적용 시도 (Pretendard)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"  # 앱에 폰트가 있으면 적용 시도
st.set_page_config(page_title="데이터 대시보드 (Streamlit + Codespaces)", layout="wide")

# 시각적 기본 스타일 (Pretendard 적용 시도 — 실패해도 무해)
st.markdown(
    f"""
    <style>
    @font-face {{
        font-family: 'PretendardCustom';
        src: local('Pretendard'), url('{PRETENDARD_PATH}') format('truetype');
        font-weight: 700;
        font-style: normal;
    }}
    html, body, [class*="css"]  {{
        font-family: PretendardCustom, Pretendard, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# 캐시된 데이터 로더들
# ----------------------------
@st.cache_data(show_spinner=False)
def fetch_worldbank_co2(start_year=1960, end_year=None, retries=2, backoff=1.0):
    """
    World Bank API에서 전세계 CO2 (per capita) 시계열을 불러옵니다.
    - start_year: 조회 시작 연도 (예: 1960)
    - end_year: 조회 종료 연도 (None이면 현재 연도로 대체)
    - 실패시 예시 데이터를 반환
    """
    if end_year is None:
        end_year = datetime.datetime.now().year  # 로컬 기준 현재 연도
    url = WORLD_BANK_API_TEMPLATE.format(indicator=WORLD_BANK_INDICATOR, start=start_year, end=end_year)
    attempt = 0
    last_error = None
    while attempt <= retries:
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                last_error = f"HTTP {resp.status_code}"
                attempt += 1
                time.sleep(backoff * attempt)
                continue
            data = resp.json()
            # World Bank API returns [metadata, data_array]
            if not isinstance(data, list) or len(data) < 2:
                last_error = "Unexpected API response format"
                attempt += 1
                time.sleep(backoff * attempt)
                continue
            records = data[1]
            rows = []
            for rec in records:
                country = rec.get("country", {}).get("value")
                country_code = rec.get("country", {}).get("id")
                year = rec.get("date")
                value = rec.get("value")
                rows.append({"country": country, "country_code": country_code, "date": year, "value": value})
            df = pd.DataFrame(rows)
            # 전처리: 형변환, 결측 처리, 중복 제거, 미래 데이터 제거
            df = df.drop_duplicates(subset=["country_code", "date"])
            df["year"] = pd.to_numeric(df["date"], errors="coerce").astype("Int64")
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            # 제거: 미래 연도 (현재 연도 초과) — API 요청시 end_year을 지정했으나 안전장치
            current_year = datetime.datetime.now().year
            df = df[df["year"].notna()]
            df = df[df["year"] <= current_year]
            df = df.sort_values(["country", "year"])
            # 표준화: date,value,group(optional)
            df_standard = df.rename(columns={"year": "date", "value": "value", "country": "group"})[["date", "value", "group", "country_code"]]
            return df_standard
        except Exception as e:
            last_error = str(e)
            attempt += 1
            time.sleep(backoff * attempt)
    # 실패 시 예시 데이터 (간단한 샘플)
    example = pd.DataFrame({
        "date": [2000, 2005, 2010, 2015, 2020],
        "value": [4.5, 5.0, 5.3, 5.1, 4.8],
        "group": ["South Korea"] * 5,
        "country_code": ["KOR"] * 5
    })
    example_notice = {
        "error": True,
        "message": f"World Bank API 호출 실패: {last_error}. 예시 데이터로 대체합니다."
    }
    # attach the notice as attribute
    example.attrs["notice"] = example_notice
    return example

@st.cache_data(show_spinner=False)
def load_user_csv_from_string(csv_text: str):
    """프롬프트에서 제공된 CSV 텍스트를 바로 DataFrame으로 변환하여 전처리 반환"""
    df = pd.read_csv(io.StringIO(csv_text))
    # 표준화: date, value, group(optional)
    # 가능한 열 매핑 시도
    df_cols = {c.lower(): c for c in df.columns}
    # 자동 매핑
    col_date = None
    col_value = None
    col_group = None
    for k, orig in df_cols.items():
        if k in ("date", "year", "년", "날짜"):
            col_date = orig
        if k in ("value", "값", "val", "value_usd", "co2"):
            col_value = orig
        if k in ("group", "country", "region", "그룹", "국가"):
            col_group = orig
    # if missing, pick heuristics
    if col_date is None:
        # try first numeric or datetime-like
        for orig in df.columns:
            if df[orig].dtype == "int64" or df[orig].dtype == "Int64":
                col_date = orig
                break
    if col_value is None:
        # pick numeric column that is not date
        for orig in df.columns:
            if pd.api.types.is_numeric_dtype(df[orig]) and orig != col_date:
                col_value = orig
                break
    # Build standardized DataFrame
    out = pd.DataFrame()
    if col_date is not None:
        out["date"] = pd.to_numeric(df[col_date], errors="coerce").astype("Int64")
    else:
        out["date"] = pd.RangeIndex(start=1, stop=len(df)+1)
    if col_value is not None:
        out["value"] = pd.to_numeric(df[col_value], errors="coerce")
    else:
        # if no numeric, try to convert first column
        out["value"] = pd.to_numeric(df.iloc[:,0], errors="coerce")
    if col_group is not None:
        out["group"] = df[col_group].astype(str)
    else:
        out["group"] = "사용자데이터"
    # remove future dates (> current year)
    current_year = datetime.datetime.now().year
    try:
        out = out[out["date"].notna()]
        out = out[out["date"] <= current_year]
    except Exception:
        pass
    out = out.drop_duplicates()
    out = out.reset_index(drop=True)
    return out

# ----------------------------
# UI: 상단 및 사이드바 공통 옵션
# ----------------------------
st.title("데이터 대시보드 — 공개 데이터 & 사용자 입력 (한국어 UI)")
st.caption("Streamlit + GitHub Codespaces 환경에서 즉시 실행 가능한 대시보드 템플릿입니다.")

# 사이드바: 공통 기능
st.sidebar.header("대시보드 설정 (공통)")
show_raw = st.sidebar.checkbox("원시 데이터 표 보기", value=False)
download_raw = st.sidebar.checkbox("전처리된 CSV 다운로드 버튼 표시", value=True)

# ----------------------------
# PART 1: 공개 데이터 대시보드 (World Bank)
# ----------------------------
st.header("1) 공개 데이터 대시보드 — World Bank: CO₂ 배출량 (1인당)")

with st.expander("데이터 로드/정보"):
    st.markdown(
        """
        **데이터 원본(예시)**:
        - World Bank (Indicator: EN.ATM.CO2E.PC) — CO₂ 배출량 (metric tons per capita)
        - API: https://api.worldbank.org/
        
        **동작 방식**:
        - World Bank API에서 자동 조회 (캐시 적용)
        - API 실패 시 예시 데이터로 대체하고 화면에 안내 표시
        - 로컬 현재 연도(시스템 시간)를 기준으로 미래 데이터는 제거
        """
    )

# 시도: World Bank 데이터 불러오기
with st.spinner("공개 데이터(World Bank) 불러오는 중..."):
    wb_df = fetch_worldbank_co2(start_year=1990, end_year=None)

# 알림: API 실패 시
if getattr(wb_df, "attrs", None) and "notice" in wb_df.attrs and wb_df.attrs["notice"].get("error", False):
    st.warning("공개 데이터 불러오기 문제: " + wb_df.attrs["notice"]["message"])

# 기본 전처리 및 사용자 제어 (공개 데이터)
if wb_df.empty:
    st.error("공개 데이터가 비어있습니다.")
else:
    # 측정 가능한 그룹(국가) 목록
    countries = wb_df["group"].dropna().unique().tolist()
    countries_display = sorted([c for c in countries if isinstance(c, str)])
    sel_country = st.sidebar.selectbox("국가 선택 (공개 데이터)", options=["World (모두)"] + countries_display, index=0)
    # 기간 필터
    years = sorted(wb_df["date"].dropna().unique().astype(int).tolist())
    if years:
        min_year, max_year = years[0], years[-1]
    else:
        min_year, max_year = 1990, datetime.datetime.now().year
    sel_year_range = st.sidebar.slider("연도 범위", min_value=int(min_year), max_value=int(max_year), value=(int(max_year)-10 if int(max_year)-10>int(min_year) else int(min_year), int(max_year)))
    smoothing = st.sidebar.slider("스무딩(이동평균, 기간)", min_value=1, max_value=5, value=1)

    # 필터 적용
    df_pub = wb_df.copy()
    df_pub = df_pub.dropna(subset=["value", "date"])
    df_pub = df_pub[(df_pub["date"] >= sel_year_range[0]) & (df_pub["date"] <= sel_year_range[1])]
    if sel_country != "World (모두)":
        df_pub = df_pub[df_pub["group"] == sel_country]

    # 집계 및 시계열 플롯
    if df_pub.empty:
        st.info("선택한 조건에 해당하는 공개 데이터가 없습니다.")
    else:
        # 시계열: date -> value
        df_pub_ts = df_pub.copy()
        # 월/연 단위: 이미 연도 단위로 정리되어 있음 (World Bank는 보통 연도)
        df_pub_ts = df_pub_ts.groupby("date").agg({"value":"mean"}).reset_index()
        if smoothing and smoothing > 1:
            df_pub_ts["value_smoothed"] = df_pub_ts["value"].rolling(window=smoothing, min_periods=1, center=False).mean()
            y_col = "value_smoothed"
            label_extra = f" — {smoothing}년 이동평균"
        else:
            y_col = "value"
            label_extra = ""
        fig = px.line(df_pub_ts, x="date", y=y_col, markers=True, title=f"{sel_country} — CO₂ 배출량 (1인당) 시계열{label_extra}",
                      labels={"date":"연도", y_col:"CO₂ 배출량 (t/person)"})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 막대: 최근 연도 비교 (최대 10개 국가 상위)
        if sel_country == "World (모두)":
            latest_year = int(df_pub["date"].max())
            df_recent = wb_df[wb_df["date"]==latest_year].dropna(subset=["value"])
            df_top10 = df_recent.sort_values("value", ascending=False).head(10)
            st.subheader(f"{latest_year}년 기준: CO₂ 배출량 (1인당) 상위 10개국")
            fig2 = px.bar(df_top10, x="group", y="value", title=f"{latest_year}년 상위 10개국 (1인당 CO₂)", labels={"group":"국가", "value":"t/person"})
            st.plotly_chart(fig2, use_container_width=True)

    # 원시 데이터 보기 / 다운로드
    if show_raw:
        st.subheader("전처리된 공개 데이터 (표준화: date, value, group, country_code)")
        st.dataframe(df_pub.reset_index(drop=True))
    if download_raw:
        csv_bytes = df_pub.to_csv(index=False).encode("utf-8")
        st.download_button("전처리 데이터 CSV 다운로드 (공개 데이터)", data=csv_bytes, file_name="public_co2_preprocessed.csv", mime="text/csv")

# ----------------------------
# PART 2: 사용자 입력 대시보드
# ----------------------------
st.markdown("---")
st.header("2) 사용자 입력 대시보드 (프롬프트 Input 기반)")

# 사용자가 실제로 본문 Input 섹션에서 CSV 텍스트를 제공했는가 검사
if USER_CSV_CONTENT and isinstance(USER_CSV_CONTENT, str) and USER_CSV_CONTENT.strip():
    st.info("프롬프트 내 제공된 사용자 CSV를 사용하여 대시보드를 생성합니다. (업로드 요구 없음)")
    try:
        user_df = load_user_csv_from_string(USER_CSV_CONTENT)
        user_notice = None
    except Exception as e:
        user_df = pd.DataFrame()
        user_notice = f"사용자 CSV 처리 중 오류: {e}"
else:
    st.warning("프롬프트의 Input 섹션에 사용자 CSV/이미지/설명이 제공되지 않았습니다. (앱은 실행 중 파일 업로드/입력 요구하지 않음) 대신 내부 예시 데이터로 사용자 대시보드를 시연합니다.")
    # 내부 예시 데이터 (시계열 예시)
    years = list(range(2010, datetime.datetime.now().year+1))
    vals_kor = [4.2,4.1,4.3,4.6,4.5,4.7,4.9,5.0,5.1,5.2,5.0,4.8,4.9,4.7,4.6,4.5]  # 길이는 years 길이에 맞춰 조정
    vals_kor = (vals_kor + [vals_kor[-1]]*len(years))[:len(years)]
    user_df = pd.DataFrame({
        "date": years,
        "value": vals_kor,
        "group": ["사용자예시_한국"] * len(years)
    })
    user_notice = "예시 데이터 사용"

# 전처리/검증
if user_df.empty:
    st.error("사용자 데이터가 비어있어 대시보드를 생성할 수 없습니다.")
else:
    # 기본 정보 및 옵션 자동 구성
    st.subheader("사용자 데이터 요약")
    st.write(f"행 개수: {len(user_df)}")
    if user_notice:
        st.info(user_notice)
    if show_raw:
        st.dataframe(user_df.head(200))

    # 자동으로 시각화 유형 결정
    # 규칙: date 컬럼이 연/월 시계열이면 시계열 플롯. group 다양성이 있으면 그룹별 비교.
    is_time_series = pd.api.types.is_integer_dtype(user_df["date"]) or pd.api.types.is_datetime64_any_dtype(user_df["date"])
    unique_groups = user_df["group"].nunique() if "group" in user_df.columns else 1

    st.sidebar.header("사용자 대시보드 옵션 (자동 구성)")
    if is_time_series:
        yr_min = int(user_df["date"].min())
        yr_max = int(user_df["date"].max())
        sel_yr_range_user = st.sidebar.slider("기간 선택 (사용자 데이터)", min_value=yr_min, max_value=yr_max, value=(max(yr_max-5, yr_min), yr_max))
    else:
        sel_yr_range_user = None
    smoothing_user = st.sidebar.slider("스무딩(이동평균) — 사용자 데이터", min_value=1, max_value=5, value=1)

    # 필터 적용
    df_user = user_df.copy()
    # 날짜 처리
    try:
        df_user["date"] = pd.to_numeric(df_user["date"], errors="coerce").astype("Int64")
    except Exception:
        pass
    if sel_yr_range_user:
        df_user = df_user[(df_user["date"] >= sel_yr_range_user[0]) & (df_user["date"] <= sel_yr_range_user[1])]
    df_user = df_user.dropna(subset=["value"])

    # 시각화 선택: 시계열이면 꺾은선/면적, 비율이면 파이/막대, 지역이면 지도 (지도은 group이 ISO코드일 때)
    st.subheader("자동 선택 시각화")
    if is_time_series:
        # 여러 그룹이면 라인 여러개, 아니면 단일 라인
        if unique_groups > 1:
            fig_user = px.line(df_user, x="date", y="value", color="group", title="사용자 데이터 — 그룹별 시계열", labels={"date":"연도", "value":"값", "group":"그룹"})
            if smoothing_user and smoothing_user > 1:
                # apply smoothing per group
                df_s = df_user.groupby("group").apply(lambda g: g.assign(value=g["value"].rolling(window=smoothing_user, min_periods=1).mean())).reset_index(drop=True)
                fig_user = px.line(df_s, x="date", y="value", color="group", title=f"사용자 데이터 — 그룹별 시계열 ({smoothing_user}년 이동평균)", labels={"date":"연도","value":"값"})
        else:
            df_user_ts = df_user.groupby("date").agg({"value":"mean"}).reset_index()
            if smoothing_user and smoothing_user > 1:
                df_user_ts["value_smoothed"] = df_user_ts["value"].rolling(window=smoothing_user, min_periods=1).mean()
                ycol = "value_smoothed"
                title_suffix = f" ({smoothing_user}년 이동평균)"
            else:
                ycol = "value"
                title_suffix = ""
            fig_user = px.area(df_user_ts, x="date", y=ycol, title=f"사용자 데이터 — 시계열{title_suffix}", labels={"date":"연도", ycol:"값"})
        st.plotly_chart(fig_user, use_container_width=True)
    else:
        # 비시계열: 값이 비율·총합이라 판단되면 파이/막대
        # 단순히 그룹별 합계를 보여줌
        df_agg = df_user.groupby("group").agg({"value":"sum"}).reset_index().sort_values("value", ascending=False)
        st.subheader("그룹별 합계 (비시계열)")
        fig_pie = px.pie(df_agg, names="group", values="value", title="사용자 데이터 — 그룹별 비율")
        st.plotly_chart(fig_pie, use_container_width=True)
        fig_bar = px.bar(df_agg, x="group", y="value", title="사용자 데이터 — 그룹별 합계", labels={"group":"그룹", "value":"값"})
        st.plotly_chart(fig_bar, use_container_width=True)

    # CSV 다운로드 (사용자 전처리 데이터)
    if download_raw:
        csv_buf = df_user.to_csv(index=False, encoding="utf-8")
        st.download_button("전처리 데이터 CSV 다운로드 (사용자 데이터)", data=csv_buf.encode("utf-8"), file_name="user_preprocessed.csv", mime="text/csv")

st.markdown("---")
st.caption("앱 노트: 이 대시보드는 Streamlit + Codespaces 환경용 템플릿입니다. World Bank API 호출 예시 및 NASA GISTEMP 정보는 코드 주석의 출처를 참고하세요.")
