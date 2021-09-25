import pickle

from PIL import Image
from pylab import *
from scipy.ndimage import filters


def get_imlist(path, fileformat='.jpg'):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(fileformat)]


def get_image_object(path):
    return Image.open(path)


def get_grayscale(imageoject):
    return imageoject.convert('L')


def convert_save(filelist):
    for infile in filelist:
        outfile = os.path.splitext(infile)[0] + ".jpg"
        if infile != outfile:
            try:
                Image.open(infile).save(outfile)
            except IOError:
                print("cannot convert", infile)


def paste(imageobject, imagetobepasted, locationtuple):
    imageobject.paste(imagetobepasted, locationtuple)
    pass


def crop(imageobject, locationtuple):
    return imageobject.crop(locationtuple)


def get_thumbnail(imageobject, sizetuple=(128, 128)):
    return imageobject.thumbnail(sizetuple)


def get_resize(imageobject, sizetuple=(128, 128)):
    return imageobject.resize(sizetuple)


def get_hist(imageobject):
    return hist(imageobject.flatten(), 128)


def get_numpyarray(imageobject):
    return array(imageobject)


def invert_im(im):
    return 255 - im


def scale_im_color_range(im, a=0, b=255):
    return uint8((b - a) / 255.0 * im + a)


def lower_value_of_dark_pixels(im):
    return uint8(255.0 * (im / 255.0) ** 2)


def get_pil(im):
    return Image.fromarray(im)


def gaussian_blur(im, standard_deviation):
    try:
        im2 = zeros(im.shape)
        for i in range(3):
            im2[:, :, i] = filters.gaussian_filter(im[:, :, i], standard_deviation)
        return uint8(im2)
    except:
        return filters.gaussian_filter(im, standard_deviation)


def prewitt_derivative(im):
    imx = zeros(im.shape)
    filters.prewitt(im, 1, imx)
    imy = zeros(im.shape)
    filters.prewitt(im, 0, imy)
    return np.reshape(np.concatenate((imx.flatten(), imy.flatten())), im.shape)


def sobel_derivative(im):
    imx = zeros(im.shape)
    filters.sobel(im, 1, imx)
    imy = zeros(im.shape)
    filters.sobel(im, 0, imy)
    return np.reshape(np.concatenate((imx.flatten(), imy.flatten())), im.shape)


def gaussian_derivative(im, standard_deviation):
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (standard_deviation, standard_deviation), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (standard_deviation, standard_deviation), (1, 0), imy)
    return np.reshape(np.concatenate((imx.flatten(), imy.flatten())), im.shape)


def ROF_denoise(im, U_init, tolerance=0.1, tau=0.125, tv_weight=100):
    m, n = im.shape  # size of noisy image
    # initialize
    U = U_init
    Px = im  # x-component to the dual field
    Py = im  # y-component of the dual field
    error = 1
    while error > tolerance:
        Uold = U
        # gradient of primal variable
        GradUx = roll(U, -1, axis=1) - U  # x-component of U's gradient
        GradUy = roll(U, -1, axis=0) - U  # y-component of U's gradient
        # update the dual varible
        PxNew = Px + (tau / tv_weight) * GradUx
        PyNew = Py + (tau / tv_weight) * GradUy
        NormNew = maximum(1, sqrt(PxNew ** 2 + PyNew ** 2))
        Px = PxNew / NormNew  # update of x-component (dual)
        Py = PyNew / NormNew  # update of y-component (dual)
        # update the primal variable
        RxPx = roll(Px, 1, axis=1)  # right x-translation of x-component
        RyPy = roll(Py, 1, axis=0)  # right y-translation of y-component
        DivP = (Px - RxPx) + (Py - RyPy)  # divergence of the dual field.
        U = im + tv_weight * DivP  # update of the primal variable
        # update of error
        error = linalg.norm(U - Uold) / sqrt(n * m)
    return U, im - U  # denoised image and texture residual


def show_image(im):
    figure()
    imshow(im)
    axis('equal')
    axis('off')
    show()


def pca(X):
    # get dimensions
    num_data, dim = X.shape
    # center data
    mean_X = X.mean(axis=0)
    X = X - mean_X
    if dim > num_data:
        # PCA - compact trick used
        M = dot(X, X.T)  # covariance matrix
        e, EV = linalg.eigh(M)  # eigenvalues and eigenvectors
        tmp = dot(X.T, EV).T  # this is the compact trick
        V = tmp[::-1]  # reverse since last eigenvectors are the ones we want
        S = sqrt(e)[::-1]  # reverse since eigenvalues are in increasing order
        for i in range(V.shape[1]):
            V[:, i] /= S
    else:
        # PCA - SVD used
        U, S, V = linalg.svd(X)
        V = V[:num_data]  # only makes sense to return the first num_data
    # return the projection matrix, the variance and the mean
    return V, S, mean_X


def pickle_write(filepath, V, immean):
    f = open(filepath, 'wb')
    pickle.dump(immean, f)
    pickle.dump(V, f)
    f.close()
    pass


def pickle_read(filepath):
    # load mean and principal components
    f = open(filepath, 'rb')
    immean = pickle.load(f)
    V = pickle.load(f)
    f.close()
    return V, immean


def sample_program():
    imlist = get_imlist(path, '.jpg')
    im = array(Image.open(imlist[0]))  # open one image to get size
    m, n = im.shape[0:2]  # get the size of the images
    # create matrix to store all flattened images
    immatrix = array([array(Image.open(im)).flatten() for im in imlist], 'f')
    # perform PCA
    V, S, immean = pca(immatrix)
    # show some images (mean and 7 first modes)
    figure()
    gray()
    subplot(2, 4, 1)
    imshow(immean.reshape(m, n))
    for i in range(7):
        subplot(2, 4, i + 2)
        imshow(V[i].reshape(m, n))
    show()
    # a numpy array im is a three dimensional object. therefore im[i,j,k] means value of color channel k in row i and column j.
    # however, expressions such as im[:2, :].sum() also make sense, as computations can be done for tuples as well.
