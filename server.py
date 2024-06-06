import streamlit as st
from Histrogram import*
from Filters import*
from Frequency_Domain import*
import matplotlib.pyplot as plt
import skimage.io as io
import seaborn as sns
sns.set_style("white")

st.set_page_config(page_title=" Image Processing", page_icon="📸", layout="wide",initial_sidebar_state="collapsed")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)


def img_figure(image,titlename):
    fig= plt.figure()        
    plt.imshow(image,cmap='gray')
    plt.axis("off")
    plt.title(titlename)
    return fig

def CDF (data):
    count, bins = np.histogram(data, bins=10)
    pdf = count / sum(count)
    cdf = np.cumsum(pdf)
    return cdf , bins

tab1, tab2, tab3  = st.tabs(["Filtering", "Histogram", "Hybrid Images" ])

with tab1:
    uploadimg1,filters = st.columns(2)
    side=st.sidebar
    uploaded_img =side.file_uploader("Upload Image",key="tab1",type={"png", "jpg", "jfif" , "jpeg"})
    add_noise=side.selectbox("Add Noise",('','Uniform_noise','Gaussian_noise','salt_pepper_noise'))
    col1,col2=side.columns(2)
    kernel_size=col1.number_input('Kernel Size',min_value=1,max_value=9, value=3,step=2)
    sigma= col2.number_input('sigma',min_value=1,max_value=10, value=1)
    col3,col4=side.columns(2)
    lower_Threshold= col3.number_input('lower Threshold',min_value=0,max_value=100, value=50)
    Higher_Threshold= col4.number_input(' Higher Threshold',min_value=50,max_value=200, value=100)
    col5,col6=side.columns(2)
    filters_option = col5.selectbox("Filters",('','Average Filter','Gaussian Filter','Median Filter', 'High Pass Filter', 'Low Pass Filter'))
    EDetection_option = col6.selectbox("Masks",('','Sobel','Roberts','Prewitt','Canny'))

    if uploaded_img is not None:
        file_path = 'Images/' +str(uploaded_img.name)
        img=grayscale(file_path)
       

        fig1= img_figure(img,"Uploaded Image")       
        uploadimg1.pyplot(fig1)  

        if add_noise== 'Uniform_noise':
            img=Uniform_noise(img)
            fig2= img_figure(img,"Uniform Noise")             
            filters.pyplot(fig2)  
        
        if add_noise== 'Gaussian_noise':
            img=Gaussian_noise(img)
            fig2= img_figure(img,"Gaussian Noise")        
            filters.pyplot(fig2)  
            
        if add_noise== 'salt_pepper_noise':
            img=salt_pepper_noise(img)
            fig2= img_figure(img,"Salt&Pepper Noise")        
            filters.pyplot(fig2)  

        if filters_option== 'Average Filter':
            filtered_img1=average_filter(img,kernel_size) 
            fig1= img_figure(filtered_img1,"Average Filter")       
            uploadimg1.pyplot(fig1)   
            
        if filters_option== 'Gaussian Filter':
            filtered_img2=Gaussian_filter(img,kernel_size,sigma)
            fig2= img_figure(filtered_img2,"Gaussian Filter")      
            uploadimg1.pyplot(fig2)  
        
        if filters_option== 'Median Filter':       
            filtered_img3=median_filter(img,kernel_size)
            fig3= img_figure(filtered_img3,"Median Filter")  
            uploadimg1.pyplot(fig3) 

        if filters_option== 'Low Pass Filter':
            # filtered_img4, H, fftshiftimg = lowPassFilter(img)
            filtered_img4 = lowpasscv2(img)
            fig4=img_figure(filtered_img4,"Low Pass Filter")         
            uploadimg1.pyplot(fig4)  

        if filters_option== 'High Pass Filter':
            # filtered_img5=highPassFilter(img)
            filtered_img5=highpasscv2(img)

            fig5= img_figure(filtered_img5,"High Pass Filter")         
            uploadimg1.pyplot(fig5)

        if EDetection_option== 'Sobel':
            Edged_img1,theta=Sobel(img) 
            fig6=img_figure(Edged_img1,"Sobel Edge")         
            filters.pyplot(fig6)   

        if EDetection_option== 'Roberts':
            Edged_img2=Roberts(img,kernel_size,sigma)
            fig7= img_figure(Edged_img2,"Roberts Edge")        
            filters.pyplot(fig7) 
        
        if EDetection_option== 'Prewitt':
            Edged_img3=Prewitt(img,kernel_size,sigma)
            fig8= img_figure(Edged_img3,"Prewitt Edge")       
            filters.pyplot(fig8)    

        if EDetection_option== 'Canny':
            Edged_img4=Canny(img,kernel_size,sigma,lower_Threshold,Higher_Threshold)
            fig9= img_figure(Edged_img4,"Canny Edge")         
            filters.pyplot(fig9)       
           
with tab2:
    uploadimg2,input_col,select_col,output_col = st.columns(4)
    uploaded_img_tab2= uploadimg2.file_uploader("Upload Image",key="tab2",type={"png", "jpg", "jfif" , "jpeg"})
    select=select_col.selectbox("Select Image",('Equalized image','Normalized image','Local Threshold image','Global Threshold image'))
    
    if uploaded_img_tab2 is not None:
        file_path = 'Images/'  +str(uploaded_img_tab2.name)
        img=cv2.imread(file_path,0)
        image = img_figure(img,"Uploaded Image")
        uploadimg2.pyplot(image)

        color_img = io.imread(file_path)
        red = color_img[:, :, 0]
        green = color_img[:, :, 1]
        blue = color_img[:, :, 2]

        cdf_red, bins_red = CDF(red)
        cdf_green, bins_green = CDF(green)
        cdf_blue, bins_blue = CDF(blue)

        figure, axs = plt.subplots(3,2)
        axs[0,0].hist(red.ravel(),bins=10,color='r',label="Red Channel")
        axs[1,0].hist(green.ravel(),bins=10,color='g',label="Green Channel")
        axs[2,0].hist(blue.ravel(),bins=10,color='b',label="Blue Channel")
        axs[0,1].plot(bins_red[1:], cdf_red,color='r', label="CDF Red")
        axs[1,1].plot(bins_green[1:], cdf_green,color='g', label="CDF Green")
        axs[2,1].plot(bins_blue[1:], cdf_blue,color='b', label="CDF Blue")
        figure.suptitle('RGB Channels')
        axs[0,0].legend()
        axs[0,1].legend()
        axs[1,0].legend()
        axs[1,1].legend()
        axs[2,0].legend()
        axs[2,1].legend()
        
        Histogram,equalized_img=histogram_fun(img) 
        normalized_img=normalize_images(img) 
        global_thres=global_threshold(img)
        local_thres=local_threshold(img)

        cdf, bins = CDF(equalized_img.ravel())

        with input_col:
            fig2=plt.figure()
            sns.distplot(Histogram, hist=True, kde=True, 
                bins=int(180/15), color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth':4})
            plt.title("Input Histogram & Distribution Curve ")
            st.pyplot(fig2)

            st.pyplot(figure) 

        with select_col:
            if select=="Equalized image":
                fig3= img_figure(equalized_img,"Equalized image")        
                st.pyplot(fig3)

            if select=="Global Threshold image":
                fig5= img_figure(global_thres,"Global Threshold image")       
                st.pyplot(fig5)

            if select=="Normalized image":
                fig4= img_figure(normalized_img,"Normalized image")      
                st.pyplot(fig4)

            if select=="Local Threshold image": 
                fig6= img_figure(local_thres,"Local Threshold image")         
                st.pyplot(fig6) 

        with output_col:
            fig7=plt.figure()
            sns.distplot(equalized_img.ravel(), hist=True, kde=True, 
                bins=int(180/15), color = 'darkblue', 
                hist_kws={'edgecolor':'black'},
                kde_kws={'linewidth':4})
            plt.title("Output Histogram & Distribution Curve ")
            st.pyplot(fig7)

            fig8=plt.figure()
            plt.plot(bins[1:],cdf,color = 'darkblue', label="Cumulative Output")
            plt.title(" Output Cumulative Curve ")
            plt.legend()
            st.pyplot(fig8)

with tab3:
    uploadimg3, uploadimg4, combinedimg = st.columns(3)
    combine_option = combinedimg.selectbox("Hybrid Images",("",'LowPass Image1 + HighPass Image2','LowPass Image2 + HighPass Image1'))
    
    uploaded_img1 = uploadimg3.file_uploader("Upload Image1",type={"png", "jpg", "jfif" , "jpeg"}, key="tab3.0")
    if uploaded_img1 is not None:
        file_path1 = 'Images/'  +str(uploaded_img1.name)
        gray_img1 = grayscale(file_path1)
        img1 = img_figure(gray_img1,"Uploaded Image1")
        uploadimg3.pyplot(img1)
        gray_img1 = np.asarray(gray_img1)
        
    uploaded_img2 = uploadimg4.file_uploader("Upload Image2",type={"png", "jpg", "jfif" , "jpeg"}, key="tab3.1")
    if uploaded_img2 is not None:
        file_path2 = 'Images/'  +str(uploaded_img2.name)
        gray_img2 = grayscale(file_path2)
        img2 = img_figure(gray_img2,"Uploaded Image2")
        uploadimg4.pyplot(img2)
        gray_img2 = np.asarray(gray_img2)
        
        with combinedimg:
            if combine_option=='LowPass Image1 + HighPass Image2':
                low_img1, H, fftshiftimg = lowPassFilter(gray_img1)
                high_img2 = highPassFilter(gray_img2)
                low1combined_img = low_img1 + high_img2
                low1combined = img_figure(low1combined_img,"") 
                st.pyplot(low1combined)

            if combine_option=='LowPass Image2 + HighPass Image1':
                low_img2, H, fftshiftimg = lowPassFilter(gray_img2)
                high_img1 = highPassFilter(gray_img1)
                low2combined_img = low_img2 + high_img1
                low2combined= img_figure(low2combined_img,"") 
                st.pyplot(low2combined)

            

