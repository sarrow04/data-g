# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import io

# 日本語フォントの文字化け対策
import japanize_matplotlib

# --- Streamlitアプリの基本設定 ---
st.set_page_config(page_title="自動EDAツール", page_icon="📈", layout="wide")
st.title("📈 自動EDA（探索的データ分析）ツール")
st.write("データの本質を、3つのステップ（全体像の把握 → 個別の深掘り → 時系列の確認）で素早く理解します。")

# --- Session Stateの初期化 ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- ヘルパー関数 ---
def create_download_button(fig, file_name, label="このグラフをダウンロード"):
    """Matplotlibのグラフオブジェクトからダウンロードボタンを生成する"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=file_name,
        mime="image/png",
    )

# --- サイドバー ---
with st.sidebar:
    st.header("1. ファイルをアップロード")
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロード", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # 日付形式の列を自動的にdatetime型として読み込むよう試みる
                df = pd.read_csv(uploaded_file, parse_dates=True)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("ファイルが正常に読み込まれました！")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --- メイン画面 ---
if st.session_state.df is not None:
    df = st.session_state.df

    # ▼▼▼ セクション1: データ全体の概要と相関分析 ▼▼▼
    st.header("セクション1: データ全体の概要と相関分析")
    with st.expander("データプレビュー、基本情報などを表示", expanded=True):
        st.subheader("データプレビュー（先頭5行）")
        st.dataframe(df.head())
        st.subheader("基本情報")
        st.markdown(f"**行数:** {df.shape[0]} 行, **列数:** {df.shape[1]} 列")
        st.subheader("欠損値の数")
        st.dataframe(df.isnull().sum().rename("欠損値の数"))

    st.subheader("【必須】全体の相関分析")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 1:
        # 相関係数（テーブル）
        st.write("▼ 相関係数")
        corr_matrix = df[numeric_cols].corr()
        st.dataframe(corr_matrix)
        
        # ヒートマップ
        st.write("▼ ヒートマップ")
        fig_corr, ax_corr = plt.subplots(figsize=(14, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax_corr)
        st.pyplot(fig_corr)
        create_download_button(fig_corr, "correlation_heatmap.png", "ヒートマップをダウンロード")
    else:
        st.info("相関分析を行うには、少なくとも2つ以上の数値列が必要です。")
    # ▲▲▲ セクション1ここまで ▲▲▲

    st.markdown("---")

    # ▼▼▼ セクション2: 列ごとの詳細分析 ▼▼▼
    st.header("セクション2: 列ごとの詳細分析")
    selected_col = st.selectbox("分析したい列を1つ選択してください", df.columns, help="列を選択すると、その列の統計量とグラフが自動で表示されます。")

    if selected_col:
        st.markdown(f"### **`{selected_col}`** 列の分析結果")

        # --- 数値データの場合 ---
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.subheader("【必須】統計量")
            st.dataframe(df[selected_col].describe())
            
            st.subheader("【必須】分布（ヒストグラムと箱ひげ図）")
            fig_dist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(df[selected_col], kde=True, ax=ax1)
            ax1.set_title(f'ヒストグラム')
            sns.boxplot(x=df[selected_col], ax=ax2)
            ax2.set_title(f'箱ひげ図')
            plt.tight_layout()
            st.pyplot(fig_dist)
            create_download_button(fig_dist, f"distribution_{selected_col}.png")

        # --- カテゴリデータの場合 ---
        else:
            st.subheader("【必須】統計量（カテゴリ別件数）")
            st.dataframe(df[selected_col].value_counts())
            
            unique_count = df[selected_col].nunique()
            if unique_count > 30:
                st.warning(f"カテゴリ数が{unique_count}と多すぎるため、グラフ描画を上位30件に制限します。")
                
            fig_count, ax_count = plt.subplots(figsize=(10, 8))
            sns.countplot(y=df[selected_col], order=df[selected_col].value_counts().nlargest(30).index, ax=ax_count)
            ax_count.set_title(f'カテゴリごとの件数（上位30件）')
            plt.tight_layout()
            st.pyplot(fig_count)
            create_download_button(fig_count, f"countplot_{selected_col}.png")
    # ▲▲▲ セクション2ここまで ▲▲▲

    st.markdown("---")

    # ▼▼▼ セクション3: 時系列データの自動グラフ化 ▼▼▼
    st.header("セクション3: 時系列データの自動グラフ化")
    datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()

    if not datetime_cols:
        st.info("データ内に日付・時刻形式の列が見つかりませんでした。CSV読み込み時に日付として認識されなかった可能性があります。")
    else:
        time_col = st.selectbox("X軸として使用する時間列を選択してください", datetime_cols)
        
        if time_col and len(numeric_cols) > 0:
            st.write(f"**`{time_col}`** を時間軸として、全ての数値データの折れ線グラフをまとめて出力します。")
            
            for num_col in numeric_cols:
                if num_col != time_col: # 時間軸自身はプロットしない
                    st.subheader(f"時系列プロット: `{num_col}`")
                    fig_line, ax_line = plt.subplots(figsize=(12, 5))
                    sns.lineplot(x=df[time_col], y=df[num_col], ax=ax_line)
                    ax_line.set_title(f'{time_col}に対する{num_col}の推移')
                    ax_line.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig_line)
                    create_download_button(fig_line, f"timeseries_{time_col}_vs_{num_col}.png")
        else:
            st.warning("グラフ化対象の数値データがありません。")
    # ▲▲▲ セクション3ここまで ▲▲▲

else:
    st.info("サイドバーからファイル（CSVまたはExcel）をアップロードして分析を開始してください。")
