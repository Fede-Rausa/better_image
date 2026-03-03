import streamlit as st
from PIL import Image
import cv2
from kalman_filter_class import kalman_rgb_img_denoiser41, kalman_rgb_img_denoiser42
from clahe_opencv import clahe_free
import numpy as np
from io import BytesIO

def cartoonify(image, ks_median=5, ks_threshold=9, ks_bilateralFilter=9, cartoon_strength=250):
    image1 = image.copy()
    g = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    g = cv2.medianBlur(g, ks_median)

    # Edges
    e = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, ks_threshold, ks_threshold)

    # Smooth color
    c = cv2.bilateralFilter(image1, ks_bilateralFilter, cartoon_strength, cartoon_strength)

    # Combine
    cartoon = cv2.bitwise_and(c, c, mask=e)

    return(cartoon)








def denoise_image(label, image, op):
    image0 = np.array(image.copy())

    if label=='enhance':
        image0 =  cv2.convertScaleAbs(image0, alpha=op['alpha'], beta=op['beta'])

    if label=='kalman':
        if op['type'] == 'rgb_cov':
            image0 = kalman_rgb_img_denoiser42(image0, R_var=op['R'], Q_var=op['Q_var'], Q_cov=op['Q_cov'], mode=op['mode'])
        else:
            image0 = kalman_rgb_img_denoiser41(image0, R=op['R'], Q=op['Q_var'], mode=op['mode'], type=op['type'])

    if label=='blur':
        if op['mode']=='gaussian':
            image0 = cv2.GaussianBlur(image0, ksize=(op['ks'], op['ks']), sigmaX=op['sigmaX'])
        else:
            image0 = cv2.medianBlur(image0, ksize=op['ks'])


    if label=='clahe':
        myclahe = clahe_free(clipLimit=op['clipLimit'], tileGridSize=(op['tileGridSize'], op['tileGridSize']), mode=op['mode'])
        image0 = myclahe.apply(image0)


    if label=='bilateral':
        image0 = cv2.bilateralFilter(image0, d=op['ks'], sigmaColor=op['strength'], sigmaSpace=op['strength'])


    if label=='cartoonify':
        image0 = cartoonify(image0, ks_median=op['ks_median'], ks_threshold=op['ks_threshold'],
                            ks_bilateralFilter=op['ks_bilateralFilter'], cartoon_strength=op['strength'])

    return image0.astype(np.uint8)


def update_sidebar_params(label, step_id):

    if label=='enhance':
        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
A simple contrast and brightness adjustment (image enhancement by hand). \n
alpha = 1 means no contrast, beta = 0 means no brightness. \n
For each pixel of each channel, its new value
is simply a linear function of its original value, 
with weight alpha and intercept beta: \n
                                
newpixel = alpha*oldpixel + beta
                                
A better alternative to this filter is clahe, for the same purposes.
            ''')
        alpha = st.sidebar.slider(label='contrast-alpha', key='alpha'+step_id, value=1.0, min_value=0.0, max_value=5.0, step=0.01,
                                  help='the contrast, or gain. That is the factor multiplied by the old value of the pixel.')
        beta = st.sidebar.slider(label='brightness-beta', key='beta'+step_id, value=0.0, min_value=-100.0, max_value=100.0, step=0.01,
                                 help='the brightness, or bias. That is the term added to the old value of the pixel multiplied by alpha.')
        return {'alpha':alpha, 'beta':beta}

    if label=='kalman':
        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
An image denoiser develop by me :) \n
It is based on Kalman filter. 
Specifically, it creates one kalman 
for multiple pixels over the same
row / column with shared parameters, 
and slides it over the image.
                                
The types I proposed are 4:

- simple: uses R (measurement noise) and Q_var (state noise)
to compute and update at each time step the kalman gain (a value between 0 and 1 that says how much we
should correct the old value of each pixel by the difference between its new value and its old value).
Approximately, You can think that K = Q / (Q+R), so is constrained in (0,1).
If R is 0, the image doesn't change. If Q is low, then the image tends to blurry.

- steady: like simple, but with a starting variance that is already converged
So the kalman gain doesn't change over the iterations.
                        
- adaptive: uses the square difference between the old state value and 
the new state value (innovation squared) as the specific Q (Q_var) used to update each pixel.
So this filter is the best for video denoising, since computes a specific K for each pixel 
and is the one more robust to blurry effects.
                                
- rgb_cov: it is the unique type that takes advantage of the covariance Q_cov.\n
Here Q_cov is the covariance between the colors of the same pixel on different channels.
So for this filte R and Q are 3x3 matrices.\n R is diagonal. \n Q = IQ_var + Q_cov - IQ_cov \n
where I is the identity 3x3, while W_var and W_cov are two scalars

         
If You select mode both,
then You create 4 kalman filters of the same type for each channel (so 4x3=12 kf in total): 
two that slides vertically over the image, in opposite directions,
and two that slides horizontally over the image.
The final prediction of each pixel channel is then the mean of the kalman filters used for the channel.
If You select mode hor, or ver, You use just 2 kalman filters per channel (2x3=6 kf in total)
that slides horizontally in opposite directions (or vertically).
           
If the type You select is rgb_cov, then all the channels are estimated by each single kalman filter,
and so You have just 4 kalman filters if the mode is both, 2 otherwise.                                

This method is similar to the gaussian kernel in speed but efficient as the bilateral filter in quality.
            ''')
        R = st.sidebar.slider(label='R', key='R'+step_id, value=400.0, min_value=0.0, max_value=100000.0, step=0.01)
        Q_var = st.sidebar.slider(label='Q', key='Q'+step_id, value=100.0, min_value=0.0, max_value=100000.0, step=0.01)
        Q_cov = st.sidebar.slider(label='Q cov', key='Qcov'+step_id, value=100.0, min_value=0.0, max_value=100000.0, step=0.01)
        mode = st.sidebar.selectbox(label='directions', key='directions'+step_id, options=['both', 'hor', 'ver'])
        type = st.sidebar.selectbox(label='method', key='method'+step_id, options=['adaptive', 'steady', 'simple', 'rgb_cov'])
        return {'R':R, 'Q_var':Q_var, 'Q_cov':Q_cov, 'mode':mode, 'type':type}

    if label=='blur':
        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
As the name suggests, this filter makes the images more blurry. Is widely used for image denoising, 
but exchanges the noise with the blurry.
It does this using a gaussian kernel (takes the mean of each patch of KxK pixels, where K is the kernel size)
or using a median kernel (takes the median of the patch). The parameter sigmaX is used only by the gaussian kernel.
                                
If You prefer quality to computational speed, check the bilateral filter.
''')
        mode=st.sidebar.selectbox(label='mode', key='mode'+step_id, options=['gaussian', 'median'])
        ks = st.sidebar.number_input(label='Kernel Size', key='Kernel Size'+step_id, value=5, min_value=1, max_value=54, step=2)
        sigmaX = st.sidebar.slider(label='Sigma X',key='Sigma X'+step_id, value=0.0, min_value=0.0, max_value=400.0, step=0.01)
        return {'ks':ks, 'sigmaX':sigmaX, 'mode':mode}
    
    if label=='clahe':
        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
This is a famous and efficient algorithm that aims to enhance images,
providing the correct level of contrast and brightness.
                                ''')
        clipLimit = st.sidebar.slider(label='clip limit', key='clip limit'+step_id, value=2, min_value=0, max_value=30, step=2)
        tileGridSize = st.sidebar.slider(label='tileGridSize', key='tileGridSize'+step_id, value=8, min_value=2, max_value=50, step=2)
        mode = st.sidebar.selectbox(label='method',key='method'+step_id, options=['lab', 'rgb', 'hsv'])
        return {'clipLimit':clipLimit, 'tileGridSize':tileGridSize, 'mode':mode}

    if label=='cartoonify':

        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
This is a filter that exploits the median filter, the bilateral filter 
and some other opencv functions to cartoonish the images.
''')
        strength = st.sidebar.slider(label='strength', key='strength'+step_id, value=250, min_value=1, max_value=255)
        ks_median = st.sidebar.number_input(label='Kernel Size Median', key='Kernel Size Median'+step_id, value=5, min_value=3, max_value=54, step=2)
        ks_threshold = st.sidebar.number_input(label='Kernel Size Threshold', key='Kernel Size Threshold'+step_id, value=9, min_value=3, max_value=54, step=2)
        ks_bilateral = st.sidebar.number_input(label='Kernel Size Bilateral', key='Kernel Size Bilateral'+step_id, value=9, min_value=3, max_value=54, step=2)
        return {'strength':strength, 'ks_median': ks_median, 'ks_bilateralFilter': ks_bilateral,'ks_threshold': ks_threshold}


    if label=='bilateral':

        helpbox = st.sidebar.checkbox(label='show help', key='help'+step_id)
        if helpbox:
            st.sidebar.markdown('''
The bilateral filter is an image denoiser much robust to blur than the gaussian and median filter.
It is used as them because is much more computationally expensive.
''')
        ks = st.sidebar.number_input(label='Distance', key='Kernel Size'+step_id, value=9, min_value=3, max_value=54, step=2)
        strength = st.sidebar.slider(label='strength', key='strength'+step_id, value=250, min_value=1, max_value=255)
        return {'strength':strength, 'ks':ks}



def theme_changer(insidebar=False):
    # Initialize theme in session state
    if 'theme' not in st.session_state:
        st.session_state.theme = 'Light☀︎'.lower()

    if insidebar:
        with st.sidebar:
            theme = st.selectbox(
                "Select Theme",
                ["Light☀︎", "Dark⏾"],
                index=0 if st.session_state.theme == 'Light☀︎'.lower() else 1
            )
    else:
        theme = st.selectbox(
            "Select Theme",
            ["Light☀︎", "Dark⏾"],
            index=0 if st.session_state.theme == 'Light☀︎'.lower() else 1
        )

    # Update theme
    if theme.lower() != st.session_state.theme:
        st.session_state.theme = theme.lower()
        #st.rerun()
        st.experimental_rerun()
    # Comprehensive dark theme CSS

    darktheme = (st.session_state.theme == 'Dark⏾'.lower())


    if darktheme:
        dark_theme_css = """
        <style>
        /* Main app background */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #0E1117;
        }
        
        /* All text elements */
        .stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div, span {
            color: #FAFAFA !important;
        }
        
        /* Selectbox main element */
        div[data-baseweb="select"] > div {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border-color: #515561 !important;
        }
        
        /* Selectbox dropdown popover */
        div[data-baseweb="popover"] {
            background-color: #262730 !important;
        }
        
        /* Selectbox dropdown menu */
        div[data-baseweb="menu"] {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        
        /* Selectbox dropdown list */
        ul[data-baseweb="menu"] {
            background-color: #262730 !important;
        }
        
        /* Selectbox dropdown items */
        li[role="option"] {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }
        
        /* Selectbox dropdown items - hover state */
        li[role="option"]:hover {
            background-color: #515561 !important;
            color: #FAFAFA !important;
        }
        
        /*Help content*/
        div[data-testid="stTooltipContent"] {
            background-color: #515561 !important;
            color: #FAFAFA !important;
        }

        /* Help icon*/
        svg[class="icon"]{
            background-color: #FAFAFA !important;
        }
        
        /* Buttons */
        .stButton button {
            background-color: #262730 !important;
            color: #FAFAFA !important;
            border: 1px solid #515561 !important;
        }
        
        .stButton button:hover {
            background-color: #515561 !important;
            color: #FAFAFA !important;
            border-color: #71757c !important;
        }
        </style>
        """
        st.markdown(dark_theme_css, unsafe_allow_html=True)



# def set_logo(path, size='large'):
#     st.set_page_config(page_icon = path)
#     st.logo(path, size = 'large')

logo_path= 'example_projects//streamlit_deploy//better_image_logo.png'

def streamlit_loop():
    st.image(logo_path, width=100)
    theme_changer(False)
    st.title("better image")
    st.subheader("An app to denoise, enhance, and apply filters to images")
    
    

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None
    original_image = Image.open(image_file)
    original_image = original_image.convert('RGB') ###really important

    available_ops = ['nothing', 'enhance','clahe', 'blur', 'bilateral', 'kalman',  'cartoonify']
    operations = []
    selected = []
    activations = []

    st.sidebar.image(logo_path, width=50)
    st.sidebar.header('Step filters')
    helpme = st.sidebar.checkbox(label='help', key='help me')
    if helpme:
        st.sidebar.markdown('''
                            **Step filters** \n
                            To use a step, first select one from the selct box, and then activate it.\n
                            Below it You will see the tuning input and the description. \n
                            The methods can be divided in two categories:\n
                            - image enhancement methods: enhancement, clahe, cartoonify\n
                            - image denoisers: blur, bilateral filter and kalman filter\n
                            Details of each algorithm are explained in its help
                            ''')
        
    Nsteps = st.sidebar.number_input('n of steps', min_value=1, value=4, max_value=20)
    for i in range(Nsteps):
        op_label = st.sidebar.selectbox('Step'+str(i+1), options=available_ops)
        is_active = st.sidebar.checkbox(label='active', key='active-Step'+str(i+1))
        op = update_sidebar_params(op_label, step_id=str(i))
        st.sidebar.text('--------------------------')
        selected.append(op_label)
        operations.append(op)
        activations.append(is_active)

    image0 = np.array(original_image.copy())  ###really important too
    for i in range(Nsteps):
        if activations[i]:
            image0 = denoise_image(label=selected[i], image=image0, op=operations[i])


    st.text("Original Image vs Processed Image")

    sidebyside = st.checkbox('show side by side', value=True)
    if sidebyside:
        col1, col2 = st.columns(2)
        col1.image(original_image)
        col2.image(image0)
    else:
        st.image(original_image)
        st.image(image0)


    if type(image0)==np.ndarray:
        image_download = Image.fromarray(image0.copy())
    else:
        image_download = image0.copy()

    # Create in-memory buffer and save image as PNG bytes
    buffer = BytesIO()
    image_download.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()

    # Download button (similar to your CSV example)
    st.download_button(
        label="Download Image",
        data=img_bytes,
        file_name="image.png",
        mime="image/png"
    )


if __name__=='__main__':
    streamlit_loop()




