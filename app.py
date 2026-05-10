import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from pipeline.skin import Srun_skin_pipeline
from pipeline.filter import Srun_filter_pipeline
from pipeline.beard import Srun_beard_pipeline
from recommend.recommend import recommend
from core.vision import white_float_score

from datatypes import FOUNDATION_DB
st.title("自分に似合うファンデを見つけよう！")


if "page" not in st.session_state:
    st.session_state.page = "home"


if st.session_state.page == "home":
    st.write("機能を選択してください")

    if st.button("肌色分析"):
        st.session_state.page = "skin"
        st.rerun()
    
    if st.button("自分で似合う色を探す"):
        st.session_state.page = "filter"
        st.rerun()

elif st.session_state.page == "skin":
    st.header("肌色分析")
    
    uploaded_file=st.file_uploader("画像をアップロード",type=["jpg","png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        max_width=600

        h,w=image.shape[:2]
        scale = max_width / w

        new_w = int(w * scale)
        new_h = int(h * scale)

        display_image = cv2.resize(image, (new_w, new_h))
        image_rgb = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        

        canvas_result=st_canvas(
            fill_color="rgba(255,0,0,0.3)",
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=pil_image,
            update_streamlit=True,
            height=new_h,
            width=new_w,
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            objects=canvas_result.json_data["objects"]

            if len(objects)>0:
                rect=objects[-1]

                scale_x = w / new_w
                scale_y = h / new_h

                x = int(rect["left"] * scale_x)
                y = int(rect["top"] * scale_y)
                w_rect = int(rect["width"] * scale_x)
                h_rect = int(rect["height"] * scale_y)
                paper_region=(x,y,x+w_rect,y+h_rect)

                st.write("選択範囲:",paper_region)

                if st.button("分析する"):
                    result = Srun_skin_pipeline(image, paper_region)
                    beard_s=Srun_beard_pipeline(result)

                    if result is None:
                        st.error("解析できませんでした")
                        st.stop()

                    st.session_state.skin_result = result
                    st.session_state.lab = result.lab
                    st.session_state.image = result.balanced_img

                    st.subheader("結果")

                        
                    st.image(result.balanced_img, channels="BGR")

                    L, a, b = result.lab
                    st.write(f"L: {L:.2f}")
                    st.write(f"a: {a:.2f}")
                    st.write(f"b: {b:.2f}")

                    st.write("青髭")
                    st.write(beard_s.lab)

                    recs=recommend(result.lab,FOUNDATION_DB)

                    st.subheader("おすすめファンデーション")

                    for name, lab, dist in recs[:3]:
                        st.write(f"{name}（距離: {dist:.2f}）")

                    best_name, best_lab, best_dist = recs[0]

                    st.write("最適ファンデ:", best_name)
                    st.write("Lab:", best_lab)
                    filtered_img,mask=Srun_filter_pipeline(result.balanced_img,result.lab,0.5)
                    st.image(filtered_img, channels="BGR")

                if  st.button("戻る"):
                        st.session_state.page = "home"
                        st.rerun()

                    
   

elif st.session_state.page == "filter":
    st.header("色調フィルター調整")

    if  st.button("戻る"):
        st.session_state.page = "home"
        st.rerun()

    if "skin_result" not in st.session_state:
        st.warning("先に肌色分析をしてください")
        st.stop()

    img = st.session_state.image
    base_lab = st.session_state.lab
    

    L = st.slider("L調整", 0.0, 100.0, float(base_lab[0]), key="L")
    A = st.slider("A調整", -128.0, 128.0, float(base_lab[1]), key="A")
    B = st.slider("B調整", -128.0, 128.0, float(base_lab[2]), key="B")
    strength = st.slider("強さ", 0.0, 1.0, 0.5, key="strength")

    lab = np.array([L, A, B])
    score=white_float_score(lab,base_lab)

  
    st.write("score=",score)
    st.write("0～10なら自然,10～25はやや白浮き,25～白浮き")
   

    
    filtered, mask =Srun_filter_pipeline(img, lab, strength)
    filtered = np.clip(filtered, 0, 255).astype(np.uint8)
    st.image(filtered, channels="BGR")
    
    