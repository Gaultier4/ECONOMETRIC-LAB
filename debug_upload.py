import streamlit as st

st.set_page_config(page_title="Debug Upload")

st.title("Debug Uploader Check")
st.write("This is a minimal script to test file uploading isolation.")

uploaded = st.file_uploader("Upload any CSV", type=["csv"])

if uploaded:
    st.success(f"File Uploaded: {uploaded.name}")
    st.info(f"Size: {uploaded.size} bytes")
    try:
        content = uploaded.getvalue()
        st.text(f"First 100 bytes: {content[:100]}")
    except Exception as e:
        st.error(f"Error reading file: {e}")
