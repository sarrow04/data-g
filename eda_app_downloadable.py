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
st.set_page_config(page_title="シンプルEDAツール", page_icon="📊", layout="wide")
st.title("📊 シンプルEDA（探索的データ分析）ツール")
st.write("データの本質を素早く、直感的に掴むためのツールです。")

# --- Session Stateの初期化 ---
if 'df' not in st.session_state:
    st.session_state.df = None

# ▼▼▼ 変更点1: グラフのダウンロードボタンを生成するヘルパー関数 ▼▼▼
def create_download_button(fig, file_name):
    """Matplotlibのグラフオブジェクトからダウンロードボタンを生成する関数"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight') # bbox_inches='tight'で余白を最適化
    st.download_button(
        label="このグラフをダウンロード",
        data=buf.getvalue(),
        file_name=file_name,
        mime="image/png",
    )
# ▲▲▲ ここまで ▲▲▲

# --- サイドバー ---
with st.sidebar:
    st.header("1. ファイルをアップロード")
    uploaded_file = st.file_uploader("CSVまたはExcelファイルをアップロード", type=['csv', 'xlsx'])

    if uploaded_file is not None:
        try:
            # 拡張子に応じて読み込み方法を変更
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("ファイルが正常に読み込まれました！")
        except Exception as e:
            st.error(f"ファイルの読み込み中にエラーが発生しました: {e}")

# --- メイン画面 ---
if st.session_state.df is not None:
    df = st.session_state.df

    # --- データ概要セクション ---
    st.header("まずはデータの全体像を把握")
    with st.expander("データプレビュー、基本統計量などを表示", expanded=True):
        st.subheader("データプレビュー（先頭5行）")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("基本情報")
            st.markdown(f"**行数:** {df.shape[0]} 行")
            st.markdown(f"**列数:** {df.shape[1]} 列")
        with col2:
            st.subheader("欠損値の数")
            st.dataframe(df.isnull().sum().rename("欠損値の数"))

        st.subheader("基本統計量")
        st.dataframe(df.describe(include='all'))

    st.markdown("---")

    # ▼▼▼ 変更点2: 新しいメインの分析セクション（タブを廃止） ▼▼▼
    st.header("詳細分析：列を選択して深掘り")
    selected_col = st.selectbox("分析したい列を1つ選択してください", df.columns)

    if selected_col:
        st.markdown(f"### **`{selected_col}`** 列の分析結果")

        # --- 数値データの場合の処理 ---
        if pd.api.types.is_numeric_dtype(df[selected_col]):
            st.subheader("データの分布")
            
            # ヒストグラムと箱ひげ図を横に並べて表示
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ヒストグラム
            sns.histplot(df[selected_col], kde=True, ax=ax1)
            ax1.set_title(f'ヒストグラム')
            
            # 箱ひげ図
            sns.boxplot(x=df[selected_col], ax=ax2)
            ax2.set_title(f'箱ひげ図')
            
            plt.tight_layout()
            st.pyplot(fig)
            create_download_button(fig, f"distribution_{selected_col}.png")

            # 他の数値変数との関係（散布図）
            with st.expander("他の数値変数との関係を見る（散布図）"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                # 自分自身は選択肢から除外
                other_numeric_cols = [col for col in numeric_cols if col != selected_col]
                if other_numeric_cols:
                    selected_scatter_col = st.selectbox("比較したいもう1つの数値列を選択", other_numeric_cols)
                    if selected_scatter_col:
                        fig_scatter, ax_scatter = plt.subplots()
                        sns.scatterplot(x=df[selected_col], y=df[selected_scatter_col], ax=ax_scatter)
                        ax_scatter.set_title(f'「{selected_col}」と「{selected_scatter_col}」の散布図')
                        st.pyplot(fig_scatter)
                        create_download_button(fig_scatter, f"scatter_{selected_col}_vs_{selected_scatter_col}.png")
                else:
                    st.info("比較対象となる他の数値列がありません。")

        # --- カテゴリデータの場合の処理 ---
        else:
            st.subheader("カテゴリごとの件数")
            
            # カテゴリ数が多すぎる場合は警告を出す
            unique_count = df[selected_col].nunique()
            st.write(f"ユニークなカテゴリ数: **{unique_count}**")

            if unique_count > 30:
                st.warning("カテゴリ数が30を超えているため、グラフの描画を停止しました。多すぎるカテゴリは分析に適さない可能性があります。")
                st.write("上位30件のカテゴリと件数を表示します:")
                st.dataframe(df[selected_col].value_counts().nlargest(30))
            else:
                fig, ax = plt.subplots(figsize=(10, max(6, unique_count * 0.4))) # カテゴリ数に応じて高さを調整
                sns.countplot(y=df[selected_col], order=df[selected_col].value_counts().index, ax=ax)
                ax.set_title(f'カテゴリごとの件数')
                plt.tight_layout()
                st.pyplot(fig)
                create_download_button(fig, f"countplot_{selected_col}.png")

            # 数値変数との関係（箱ひげ図）
            with st.expander("数値変数との関係を見る（カテゴリ別 箱ひげ図）"):
                numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not numeric_cols:
                    st.info("比較対象となる数値列がありません。")
                elif unique_count > 30:
                     st.warning("カテゴリ数が多すぎるため、箱ひげ図の描画はできません。")
                else:
                    selected_box_num_col = st.selectbox("比較したい数値列を選択", numeric_cols)
                    if selected_box_num_col:
                        fig_box, ax_box = plt.subplots(figsize=(10, max(6, unique_count * 0.4)))
                        sns.boxplot(x=df[selected_box_num_col], y=df[selected_col], ax=ax_box)
                        ax_box.set_title(f'「{selected_col}」ごとの「{selected_box_num_col}」の分布')
                        plt.tight_layout()
                        st.pyplot(fig_box)
                        create_download_button(fig_box, f"boxplot_{selected_col}_vs_{selected_box_num_col}.png")
    # ▲▲▲ ここまでが新しい分析セクション ▲▲▲

else:
    st.info("サイドバーからファイル（CSVまたはExcel）をアップロードして分析を開始してください。")
