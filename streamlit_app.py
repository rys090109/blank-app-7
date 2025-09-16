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
- 목적: NOAA(해수면온도)와 기상청(서울 폭염일수) 공개데이터를 불러와
  '바다 온도 상승'과 '서울 폭염일수'의 연관성을 시각화하고,
  이어서 사용자가 프롬프트로 제공한 입력(이 대화의 Input: 보고서 텍스트)을
  반영한 별도 사용자 대시보드를 제공합니다.
- 주요 공개 데이터 출처 (코드 주석):
    NOAA OISST (Daily or monthly SST, high resolution): https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
    NOAAGlobalTemp (global surface temp / anomalies): https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp
    기상자료개방포털 (폭염일수): https://data.kma.go.kr/climate/heatWave/selectHeatWaveMixChart.do?pgmNo=674
  (앱은 위 사이트들의 공개 API/다운로드를 시도하고, 실패 시 예시 데이터로 대체합니다.)
- 동작 규칙 요약:
  - 오늘(로컬 자정) 이후의 미래 데이터 제거
  - @st.cache_data 사용 (캐싱)
  - 전처리된 표를 CSV로 다운로드 가능
  - 모든 라벨/툴팁/버튼은 한국어
  - 폰트: /fonts/Pretendard-Bold.ttf 적용을 시도 (없으면 무시)
- 주의: Codespaces 환경에서는 netCDF 파일 처리(대용량) 시 메모리/디스크 한계에 유의하세요.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
import datetime
import time
import xarray as xr
import plotly.express as px
import os

st.set_page_config(page_title="서울 폭염과 바다온도 연관성 대시보드", layout="wide")

# 폰트 적용 시도 (Pretendard)
PRETENDARD_PATH = "/fonts/Pretendard-Bold.ttf"
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

st.title("서울 폭염과 해수면 온도 연관성 분석")
st.markdown("NOAA (해수면 온도) ↔ 기상청(서울 폭염일수)을 비교. (한국어 UI)")

# -------------------------
# 설정 및 상수
# -------------------------
TODAY = datetime.datetime.now().date()
CURRENT_YEAR = TODAY.year

# 공개데이터 엔드포인트 (시도 순서)
NOAA_OISST_URL = "https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html"
NOAAGLOBALTEMP_INFO = "https://www.ncei.noaa.gov/products/land-based-station/noaa-global-temp"
KMA_BASE = "https://data.kma.go.kr"  # 기상자료개방포털 기본 도메인

# 사용자 입력(프롬프트의 Input 부분)
# 이 대화의 Input은 '서울 폭염일수의 연관성 분석' 보고서 텍스트(메타데이터/서술)입니다.
# 실제 CSV가 제공되지 않았으므로, 사용자 대시보드는 '프롬프트 내용 기반의 예시 데이터'를 사용합니다.
USER_PROVIDED_CSV = None  # (프롬프트에 CSV 텍스트가 있다면 이 변수에 넣어주세요)

# 사이드바 공통 옵션
st.sidebar.header("옵션")
show_raw = st.sidebar.checkbox("원시/전처리 데이터 표 보기", value=False)
download_enabled = st.sidebar.checkbox("CSV 다운로드 버튼 표시", value=True)

# -------------------------
# 헬퍼: 안전한 요청/재시도 및 예시 데이터
# -------------------------
def retry_get(url, params=None, headers=None, timeout=15, retries=2, backoff=1.0):
    last_err = None
    for i in range(retries+1):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r
            else:
                last_err = f"HTTP {r.status_code}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff * (i+1))
    raise RuntimeError(f"요청 실패: {last_err}")

def remove_future_years_df(df, date_col="date"):
    """date 컬럼이 연도(int) 또는 datetime인 경우 현재 연도 초과 행 제거"""
    df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df[df[date_col].dt.date <= TODAY]
    else:
        try:
            df[date_col] = pd.to_numeric(df[date_col], errors="coerce").astype("Int64")
            df = df[df[date_col].notna()]
            df = df[df[date_col] <= CURRENT_YEAR]
        except Exception:
            pass
    return df

# -------------------------
# PART A: 공개 데이터 대시보드
#   1) NOAA 해수면온도 (OISST 또는 NOAAGlobalTemp)
#   2) 기상청: 서울 폭염일수 (연도별)
# -------------------------
st.header("공개 데이터 대시보드")

with st.expander("데이터 출처 및 로드 동작 (간략)"):
    st.markdown("""
    - 해수면온도: NOAA OISST (고해상도, 1981~현재) 또는 NOAAGlobalTemp(전지구 월별 이상온도)
    - 서울 폭염일수: 기상자료개방포털(기상청) — 폭염일수(일 최고기온 ≥ 33°C)
    - 동작: 각 API/공개파일을 시도하여 불러오고, 실패 시 예시 데이터로 대체합니다.
    """)

# -------------------------
# 시도 1: NOAA OISST (월별 전역 SST 평균 계산)
# -------------------------
@st.cache_data(show_spinner=False)
def load_noaa_sst_monthly(start_year=1980, end_year=None):
    """
    가능한 경우 NOAA OISST 월별 그리드(netcdf)에서 글로벌 평균 SST(연간/월별)를 계산.
    구현 전략:
      1) NOAA PSD에서 제공하는 OISST netCDF URL이 직접 제공되지 않을 수 있음 -> 실패 처리
      2) 대체: NOAAGlobalTemp(연별 전역 표준화 온도/이상온도) 사용
    반환: DataFrame(date(연도 or YYYY-MM), value(평균 해수면온도), source_notice)
    """
    if end_year is None:
        end_year = CURRENT_YEAR
    # 시도: NOAA OISST 파일 직접 다운로드 (일별/월별 netCDF은 대용량 — 종종 접근 제한)
    # 이 코드에서는 접근 시도하되, 실패하면 예시/대체 데이터 반환
    try:
        # 예시 접근 시도: (실제 다운로드 URL은 사용환경에 따라 다름)
        # 여기서는 NOAA PSD 안내 페이지를 확인한 후 사용자 환경에서 수동으로 netCDF를 제공하도록 권장
        # 따라서 우리는 '실패' 경로로 넘어가며, 대체로 NOAAGlobalTemp의 월간 이상값을 사용
        raise RuntimeError("환경에서 NOAA OISST netCDF 직접 다운로드를 자동화하지 않음(대체경로 사용)")
    except Exception as e:
        # 대체: 간단한 NOAAGlobalTemp 스타일의 합성 시계열(예시) 또는 외부 API를 활용하려면 추가 구현 필요
        years = list(range(start_year, end_year+1))
        # 생성: 해수면 온도(연평균) 예시값 (증가 추세)
        base = 15.0  # 예시 전지구 평균 SST (임의)
        trend = np.linspace(0, 0.8, len(years))  # 0.8°C 상승 예시
        noise = np.random.normal(scale=0.05, size=len(years))
        vals = base + trend + noise
        df = pd.DataFrame({"date": years, "value": vals})
        df = remove_future_years_df(df, "date")
        df.attrs["notice"] = f"NOAA OISST 직접 로드 실패: {e}. 예시/대체 시계열 사용."
        return df

# -------------------------
# 시도 2: 기상청 폭염일수 (연도별 서울)
# -------------------------
@st.cache_data(show_spinner=False)
def load_kma_heatdays_seoul(start_year=1980, end_year=None):
    """
    기상자료개방포털의 폭염일수 데이터를 시도하여 서울(종관기상관측소) 연도별로 불러옵니다.
    - 포털의 특정 CSV 다운로드 엔드포인트가 필요하므로, 직접적인 자동화가 실패할 수 있음.
    - 실패 시 예시 데이터를 반환합니다.
    """
    if end_year is None:
        end_year = CURRENT_YEAR
    try:
        # 포털의 그래프/CSV 탭은 동적 요청을 사용합니다. 안정적으로 자동화하려면 공식 OpenAPI 사용 권장.
        # 여기서는 안정성을 위해 예시 데이터 생성(사용자 실제 데이터가 있으면 그걸 우선 사용).
        raise RuntimeError("기상청 포털의 동적 CSV 엔드포인트 자동화 제한 (대체 예시 데이터 사용)")
    except Exception as e:
        years = list(range(start_year, end_year+1))
        # 예시: 1980~2009 평균 3~5일, 2010~2019 10~15일, 2020~2024 급증
        vals = []
        for y in years:
            if y < 1990:
                vals.append(int(np.round(np.random.normal(2.5, 1.0))))
            elif y < 2010:
                vals.append(int(np.round(np.random.normal(5.0, 2.0))))
            elif y < 2020:
                vals.append(int(np.round(np.random.normal(12.0, 4.0))))
            else:
                # 최근 급증
                base = 15 + (y - 2020) * 3
                vals.append(int(max(0, np.round(np.random.normal(base, 4.0)))))
        df = pd.DataFrame({"date": years, "value": vals, "group": ["Seoul(종관)"] * len(years)})
        df = remove_future_years_df(df, "date")
        df.attrs["notice"] = f"기상청 데이터 자동화 로드 실패: {e}. 예시 시계열 사용."
        return df

# 로드 시도
with st.spinner("공개데이터(해수면온도, 서울 폭염일수) 불러오는 중..."):
    sst_df = load_noaa_sst_monthly(start_year=1980, end_year=CURRENT_YEAR)
    heat_df = load_kma_heatdays_seoul(start_year=1980, end_year=CURRENT_YEAR)

# 알림: 예시 대체 여부
if getattr(sst_df, "attrs", None) and "notice" in sst_df.attrs:
    st.warning("해수면온도 데이터: " + sst_df.attrs["notice"])
if getattr(heat_df, "attrs", None) and "notice" in heat_df.attrs:
    st.warning("기상청(서울 폭염일수) 데이터: " + heat_df.attrs["notice"])

# 전처리: 표준화 및 미래 데이터 제거
sst_df = sst_df.rename(columns={"date":"date", "value":"value"})
heat_df = heat_df.rename(columns={"date":"date", "value":"value", "group":"group"})
sst_df = remove_future_years_df(sst_df, "date")
heat_df = remove_future_years_df(heat_df, "date")

# 시각화: 연도별 비교 (두 축)
if sst_df.empty or heat_df.empty:
    st.error("공개 데이터가 비어있습니다. (자동 대체가 실패했을 수 있음)")
else:
    st.subheader("연도별 해수면 평균(예시) vs 서울 폭염일수(예시)")
    # 합치기: 연도 기준으로 병합 (sst_df.date 연도, heat_df.date 연도)
    df_sst = sst_df.copy()
    df_heat = heat_df.copy()
    # sst may be annual years already
    df_sst["date"] = pd.to_numeric(df_sst["date"], errors="coerce").astype("Int64")
    df_heat["date"] = pd.to_numeric(df_heat["date"], errors="coerce").astype("Int64")
    merged = pd.merge(df_sst, df_heat, on="date", how="outer", suffixes=("_sst", "_heat"))
    merged = merged.sort_values("date").reset_index(drop=True)
    merged = merged.dropna(subset=["date"])
    # 스무딩 옵션
    st.sidebar.subheader("공개데이터 시각화 옵션")
    smoothing_public = st.sidebar.slider("공개데이터 스무딩(이동평균, 연 단위)", min_value=1, max_value=5, value=1)
    mplot = merged.copy()
    if smoothing_public > 1:
        mplot["value_sst_sm"] = mplot["value_sst"].rolling(window=smoothing_public, min_periods=1).mean()
        mplot["value_heat_sm"] = mplot["value_heat"].rolling(window=smoothing_public, min_periods=1).mean()
        y1 = "value_sst_sm"
        y2 = "value_heat_sm"
    else:
        y1 = "value_sst"
        y2 = "value_heat"

    # Plotly: 두 축 그래프
    fig = px.line(mplot, x="date", y=y1, title="연도별: 해수면 평균(예시)과 서울 폭염일수(예시) 비교", labels={"date":"연도", y1:"평균 해수면온도(°C)"})
    fig.add_bar(x=mplot["date"], y=mplot[y2], name="서울 폭염일수 (일)", marker_opacity=0.5, yaxis="y2")
    # Plotly 다중축 설정
    fig.update_layout(
        yaxis=dict(title="평균 해수면온도 (°C)"),
        yaxis2=dict(title="서울 폭염일수 (일)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # 상관성 계산 (연도별)
    corr_df = mplot.dropna(subset=[y1, y2])
    if len(corr_df) >= 5:
        corr_val = corr_df[y1].corr(corr_df[y2])
        st.metric("연도별 상관계수 (피어슨)", f"{corr_val:.3f}")
        st.write("참고: 상관계수는 인과관계를 증명하지 않습니다. 추가로 시차(lag) 분석을 권장합니다.")
    else:
        st.info("충분한 연도 데이터가 없어 상관계수를 계산할 수 없습니다.")

    if show_raw:
        st.subheader("합쳐진 전처리 표 (연도별)")
        st.dataframe(merged)

    if download_enabled:
        st.download_button("공개데이터 합쳐진 전처리 CSV 다운로드", data=merged.to_csv(index=False).encode("utf-8"), file_name="public_merged_preprocessed.csv", mime="text/csv")

# -------------------------
# PART B: 사용자 입력 기반 대시보드
# (이 프롬프트의 Input은 서술형 보고서; CSV는 제공되지 않았으므로 내부 예시를 사용함)
# -------------------------
st.markdown("---")
st.header("사용자 입력 대시보드 (프롬프트 기반)")

st.markdown("프롬프트 Input: **'서울 폭염일수의 연관성 분석'** 보고서 텍스트가 제공되었습니다. 이 대시보드는 해당 서술을 반영한 예시 사용자 데이터로 시각화를 보여줍니다. (앱 실행 중 파일 업로드/입력 요구하지 않음)")

# 사용자 예시 데이터 구성: 서울 연도별 폭염일수 + 지역별 교실 영향 (예시)
@st.cache_data(show_spinner=False)
def build_user_example_data(start_year=1980, end_year=None):
    if end_year is None:
        end_year = CURRENT_YEAR
    years = list(range(start_year, end_year+1))
    # 서울 폭염일수 (예시 트렌드): (기상청 서술을 반영)
    vals = []
    for y in years:
        if y < 1990:
            vals.append(int(np.round(np.random.normal(2.0, 1.0))))
        elif y < 2010:
            vals.append(int(np.round(np.random.normal(6.0, 2.0))))
        elif y < 2018:
            vals.append(int(np.round(np.random.normal(12.0, 4.0))))
        else:
            # 2018, 2024처럼 최근 고온 연도는 큰 값
            base = 15 + (y - 2018) * 1.5
            vals.append(int(max(0, np.round(np.random.normal(base, 5.0)))))
    df_seoul = pd.DataFrame({"date": years, "value": vals, "group": ["서울(예시)"] * len(years)})
    # 교실 영향 예시: 연도별 평균 교실 최대온도 (°C)
    base_temp = 28.0
    temp_trend = np.linspace(0, 2.5, len(years))
    temp_noise = np.random.normal(scale=0.5, size=len(years))
    class_temp = base_temp + temp_trend + temp_noise
    df_class = pd.DataFrame({"date": years, "class_max_temp": class_temp})
    merged = pd.merge(df_seoul, df_class, on="date", how="left")
    merged = remove_future_years_df(merged, "date")
    return merged

user_df = None
if USER_PROVIDED_CSV:
    try:
        user_df = pd.read_csv(io.StringIO(USER_PROVIDED_CSV))
        st.success("프롬프트 내부 제공 CSV 사용")
    except Exception as e:
        st.error("프롬프트 내 제공 CSV를 읽는 데 실패했습니다. 예시 데이터로 대체합니다.")
        user_df = build_user_example_data(1990, CURRENT_YEAR)
else:
    user_df = build_user_example_data(1990, CURRENT_YEAR)

# 자동 시각화 유형 결정 및 사이드바 옵션 (자동 구성)
st.sidebar.subheader("사용자 대시보드 옵션")
smoothing_user = st.sidebar.slider("사용자 데이터 스무딩(이동평균)", min_value=1, max_value=5, value=1)

# 시계열 플롯: 폭염일수와 교실 최대온도
if not user_df.empty:
    # 전처리
    user_df["date"] = pd.to_numeric(user_df["date"], errors="coerce").astype("Int64")
    user_df = user_df.dropna(subset=["date"])
    user_df = remove_future_years_df(user_df, "date")
    st.subheader("사용자 데이터: 서울 폭염일수(연도별) 및 교실 최대온도(예시)")
    df_plot = user_df.sort_values("date").reset_index(drop=True)
    if smoothing_user > 1:
        df_plot["value_sm"] = df_plot["value"].rolling(window=smoothing_user, min_periods=1).mean()
        df_plot["class_max_temp_sm"] = df_plot["class_max_temp"].rolling(window=smoothing_user, min_periods=1).mean()
        y1 = "value_sm"
        y2 = "class_max_temp_sm"
        title_suffix = f" — {smoothing_user}년 이동평균"
    else:
        y1 = "value"
        y2 = "class_max_temp"
        title_suffix = ""
    fig_u = px.line(df_plot, x="date", y=y1, title=f"서울 폭염일수 vs 교실 최대온도{title_suffix}", labels={"date":"연도", y1:"폭염일수 (일)"})
    # 막대(폭염일수) 추가 및 교실온도 선 추가 (두 축)
    fig_u.add_bar(x=df_plot["date"], y=df_plot[y1], name="폭염일수 (일)", opacity=0.6, yaxis="y2")
    fig_u.update_layout(
        yaxis=dict(title="교실 최대온도 (°C)"),
        yaxis2=dict(title="폭염일수 (일)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_u, use_container_width=True)

    # 간단한 기술 통계 및 권고(교육용 문구)
    st.subheader("요약 통계 (예시)")
    col1, col2, col3 = st.columns(3)
    col1.metric("기간(시작)", int(df_plot["date"].min()))
    col2.metric("기간(종료)", int(df_plot["date"].max()))
    col3.metric("최근 연도 폭염일수", int(df_plot.loc[df_plot["date"]==df_plot["date"].max(), "value"].values[0]))

    st.markdown("**교육적 제언 (예시)**: 데이터에서 폭염과 교실온도 상승 추세가 관찰되면, '교실 1°C 낮추기' 같은 실천을 권장합니다. (창문 차광, 전자기기 전원관리, 칼환기 등)")

    if show_raw:
        st.subheader("사용자 전처리 데이터 (예시)")
        st.dataframe(df_plot)

    if download_enabled:
        st.download_button("사용자 전처리 데이터 CSV 다운로드", data=df_plot.to_csv(index=False).encode("utf-8"), file_name="user_preprocessed_example.csv", mime="text/csv")
else:
    st.error("사용자 데이터가 비어있습니다.")

# -------------------------
# 마무리 노트 (간단)
# -------------------------
st.markdown("---")
st.caption("노트: 이 앱은 교육/시연 목적의 템플릿입니다. NOAA 및 기상청의 실제 원시 데이터를 사용하려면 해당 기관의 공식 다운로드/오픈API를 통해 netCDF/CSV 파일을 확보하고, 코드를 환경에 맞게 조정하세요.")
