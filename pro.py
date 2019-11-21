import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
def fourier(img):
    dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return dft_shift;
def inv_fourier(img):
    f_ishift = np.fft.ifftshift(img)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back
def magnitude(dft_shift):
    return 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]));
def gabor_fn(sigma, theta, Lambda, psi, gamma):
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation 
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb
img=cv2.imread('input_image.jpg')
imgl=cv2.cvtColor(img,cv2.COLOR_BGR2LAB);
fl=imgl[:,:,0];
fa=imgl[:,:,1];
fb=imgl[:,:,2];
plt.figure()
plt.title("f_L");
plt.imshow(fl,cmap='gray');
plt.figure()
plt.title("f_a");
plt.imshow(fa,cmap='gray');
plt.figure()
plt.title("f_b");
plt.imshow(fb,cmap='gray');

gabor_f=np.zeros((fl.shape[0],fl.shape[1],2));
w_o=0.002;
sigma_f=6.2
xc=fl.shape[0]/2;
yc=fl.shape[1]/2;
for x in range(1,img.shape[0]+1):
    for y in range(1,img.shape[1]+1):
        l2_norm=max(1,math.sqrt((x-xc)*(x-xc)+(y-yc)*(y-yc)));
        #print(l2_norm/w_o);
        gabor_f[x-1,y-1]=math.exp(-(math.log(l2_norm/w_o)**2)/(2*(sigma_f**2)));
        
plt.figure()
plt.title("Gabor fiter")
plt.imshow(gabor_f[:,:,0],cmap='gray')


res_fl=np.multiply(gabor_f,fourier(fl/255.0));
#print(res_fl.shape)
res_fa=np.multiply(gabor_f,fourier(fa/255.0));
res_fb=np.multiply(gabor_f,fourier(fb/255.0));

fl_p = inv_fourier(res_fl)
#print(fl_p)
#fl_p = cv2.magnitude(fl_p[:,:,0],fl_p[:,:,1])
fa_p = inv_fourier(res_fa)
#fa_p = cv2.magnitude(fa_p[:,:,0],fa_p[:,:,1])
fb_p = inv_fourier(res_fb)
#fb_p = cv2.magnitude(fb_p[:,:,0],fb_p[:,:,1])
plt.figure()
plt.title("f_l*g(x)")
plt.imshow(fl_p,cmap='gray')
plt.figure()
plt.title("f_a*g(x)")
plt.imshow(fa_p,cmap='gray')
plt.figure()
plt.title("f_b*g(x)")
plt.imshow(fb_p,cmap='gray')
S_f=np.sqrt(np.multiply(fl_p/255.0,fl_p/255.0)+np.multiply(fa_p/255.0,fa_p/255.0)+np.multiply(fb_p/255.0,fb_p/255.0));
fa_min=fa.min();
fa_max=fa.max();
fb_min=fb.min();
fb_max=fb.max();
f_an=((np.float32(fa)-fa_min))/float(fa_max-fa_min)
f_bn=(np.float32(fb-fb_min))/float(fb_max-fb_min)
sigma_c=0.25;
S_c=1.0-np.exp(-((f_an*f_an)+(f_bn*f_bn))/(sigma_c**2));
xc=img.shape[0]/2;
yc=img.shape[1]/2;
sigma_d=114
s_d=np.zeros(fa.shape,dtype=np.float32);
for x in range(img.shape[0]):
    for y in range(img.shape[1]):
        l2_norm=float((x-xc)**2+(y-yc)**2)/(img.shape[0]*img.shape[1]);
        #print(l2_norm)
        s_d[x,y]=math.exp(-(l2_norm)/float(sigma_d**2));
print(S_f)
sdsp=S_f*S_c*s_d;
plt.figure()
plt.title("S_f")
plt.imshow(S_f,cmap='gray');
plt.figure()
plt.title("S_c")
plt.imshow(S_c,cmap='gray');
plt.figure()
plt.title("S_d")
plt.imshow(s_d,cmap='gray');
plt.figure()
plt.title("SDSP")

plt.imshow(sdsp,cmap='gray');
plt.show()
#sdsp=sdsp
#Ta=(2.0/(sdsp.shape[0]*sdsp.shape[1]))*(255*sdsp.sum())
#print(sdsp/255.0)
#for x in range(sdsp.shape[0]):
#    for y in range(sdsp.shape[1]):
#        if(sdsp[x,y]<Ta):
#            sdsp[x,y]=0;
#plt.figure()
#plt.imshow(sdsp,cmap='gray')

