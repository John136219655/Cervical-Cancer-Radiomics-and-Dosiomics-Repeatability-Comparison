import os
from math import ceil
import pandas as pd
import numpy as np
import SimpleITK as sitk
import cv2
import matplotlib.cm as mc
import scipy
import skimage.filters as sf
import scipy.ndimage as sn
import scipy.spatial
from scipy.interpolate import Rbf, griddata
def mask_to_bounding_box(mask_image:sitk.Image, margin_size=5):
    origin = mask_image.GetOrigin()
    resolution = mask_image.GetSpacing()
    size = mask_image.GetSize()
    mask = sitk.GetArrayFromImage(mask_image)
    coordinates = []
    for i in range(3):
        coordinates.append(np.arange(size[i])*resolution[i]+origin[i])
    boundary_index = np.array(
        [[len(coordinates[2]) - 1, 0], [len(coordinates[1]) - 1, 0], [len(coordinates[0]) - 1, 0]])
    for i in range(len(coordinates)):
        for j in range(mask.shape[i]):
            slice_included = np.any(np.take(mask, j, axis=i))
            if slice_included:
                if j < boundary_index[i, 0]:
                    boundary_index[i, 0] = j
                if j > boundary_index[i, 1]:
                    boundary_index[i, 1] = j
    mask_boundary = [[coordinates[0][boundary_index[2, 0]], coordinates[0][boundary_index[2, 1]]],
                     [coordinates[1][boundary_index[1, 0]], coordinates[1][boundary_index[1, 1]]],
                     [coordinates[2][boundary_index[0, 0]], coordinates[2][boundary_index[0, 1]]]]
    for i in range(len(mask_boundary)):
        mask_boundary[i][0] = max(mask_boundary[i][0], coordinates[i][0])-margin_size
        mask_boundary[i][1] = min(mask_boundary[i][1], coordinates[i][-1])+margin_size
    return mask_boundary


def crop_mask(mask_image:sitk.Image, margin_size=5, allow_free_rotation=False, reduce_slice=False):
    bounding_box = mask_to_bounding_box(mask_image, margin_size=margin_size)
    spacing = mask_image.GetSpacing()
    if allow_free_rotation:
        diagonal_distance = ((bounding_box[0][1]-bounding_box[0][0])**2+(bounding_box[1][1]-bounding_box[1][0])**2)**0.5
        box_length = diagonal_distance/np.sqrt(2)
        center = [int((position[1]+position[0])/2) for position in bounding_box]
        origin = [center[0]-box_length/2,center[1]-box_length/2,bounding_box[2][0]]
        size = [ceil(box_length/spacing[0]),ceil(box_length/spacing[1]),ceil((bounding_box[2][1]-bounding_box[2][0])/spacing[2])]
    else:
        origin = [position[0] for position in bounding_box]

        size = [ceil((position[1]-position[0])/spacing_1d) for position, spacing_1d in zip(bounding_box, spacing)]

        if reduce_slice:
            origin[2] = origin[2] + (bounding_box[2][1] - bounding_box[2][0]) / 2 - 20
            size[2] = 8
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputOrigin(origin)
    resampler.SetSize(size)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputSpacing(spacing)
    cropped_image = resampler.Execute(mask_image)
    return cropped_image

def align_masks(masks, spacing=None):
    if len(masks) == 0:
        return masks
    origins = []
    absolute_sizes = []
    spacings = []
    for mask in masks:
        origins.append(mask.GetOrigin())
        absolute_sizes.append(np.array(mask.GetSize())*np.array(mask.GetSpacing()))
        spacings.append(mask.GetSpacing())
    origin = np.min(origins, axis=0).tolist()
    if spacing is None:
        spacing = np.min(spacings, axis=0).tolist()
    size = np.ceil(np.max(absolute_sizes, axis=0)/spacing).astype(int).tolist()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    # binary_image_filter = sitk.ThresholdImageFilter()
    # binary_image_filter.SetLower(0.5)
    aligned_masks = [resampler.Execute(mask) for mask in masks]
    return aligned_masks

def draw_randomized_field(image, mask, export_directory, times=1):
    randomization_intensity = np.array([0,1,1])
    smoothing_sigma = np.array([5,5,5])
    # color_map = mc.get_cmap('rainbow')
    # mask = crop_mask(mask, margin_size=20, allow_free_rotation=True)
    # min_max_image_filter = sitk.MinimumMaximumImageFilter()
    # min_max_image_filter.Execute(image)
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetOutputOrigin(mask.GetOrigin())
    # resampler.SetOutputSpacing(mask.GetSpacing())
    # resampler.SetSize(mask.GetSize())
    # resampler.SetDefaultPixelValue(min_max_image_filter.GetMinimum())
    # image = resampler.Execute(image)
    image_origin = np.array(image.GetOrigin())
    image_spacing = np.array(image.GetSpacing())
    image_size = np.array(image.GetSize())

    downsampled_image_origin = image_origin-2*smoothing_sigma
    downsample_image_spacing = image_spacing*smoothing_sigma
    downsample_image_size = np.ceil((image_size*image_spacing+4*smoothing_sigma)/downsample_image_spacing).astype(int).tolist()
    down_sampler = sitk.ResampleImageFilter()
    down_sampler.SetOutputOrigin(downsampled_image_origin)
    down_sampler.SetOutputSpacing(downsample_image_spacing)
    down_sampler.SetSize(downsample_image_size)

    upsampler = sitk.ResampleImageFilter()
    upsampler.SetOutputOrigin(image_origin)
    upsampler.SetOutputSpacing(image_spacing)
    upsampler.SetSize(image.GetSize())

    # sitk.WriteImage(mask,os.path.join(export_directory, 'original_mask.mha'))
    # sitk.WriteImage(image, os.path.join(export_directory, 'image.mha'))
    image_array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(mask)

    dimensions = mask_array.shape
    randomized_masks = [mask]
    coordinates = np.meshgrid(range(dimensions[0]), range(dimensions[1]), range(dimensions[2]), indexing='ij')
    for time in range(times):
        time_export_directory = os.path.join(export_directory, 'time{0}'.format(time))
        if not os.path.exists(time_export_directory):
            os.mkdir(time_export_directory)
        random_smoothed_field_3d = []
        warpped_coordinates = []
        # random_state_outer = np.random.RandomState(seed=random_seed)
        for d in range(len(dimensions)):
            # random_state_inner = np.random.RandomState()
            # image_edge_single_dimension = image_edge_single_dimensions[d]
            random_state_inner = np.random.RandomState()
            if d == 0:
                random_field = random_state_inner.uniform(-1, 1, size=dimensions[0])
                random_field = np.array([np.ones([dimensions[1], dimensions[2]]) * z for z in random_field])
            else:
                random_field = random_state_inner.uniform(-1, 1, size=dimensions)
            random_field = sf.gaussian(random_field, smoothing_sigma[d], mode='wrap')
            random_field = random_field / np.linalg.norm(random_field.flatten()) * np.sqrt(
                np.prod(random_field.shape)) * randomization_intensity[d]
            random_smoothed_field_3d.append(random_field)
            warpped_coordinates.append(coordinates[d] + random_field)
        random_smoothed_field_3d = np.array(random_smoothed_field_3d)
        random_smoothed_field_x_y = random_smoothed_field_3d[[2, 1], int(random_smoothed_field_3d.shape[1] / 2), :, :]
        random_smoothed_field_y_z = random_smoothed_field_3d[[1, 0], :, :, int(random_smoothed_field_3d.shape[3] / 2)]
        random_smoothed_field_x_z = random_smoothed_field_3d[[2, 0], :, int(random_smoothed_field_3d.shape[2] / 2), :]
        pd.DataFrame(random_smoothed_field_x_y[0, ::8, ::8]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_xy_x.csv'), index=None,
                                                                header=None)
        pd.DataFrame(random_smoothed_field_x_y[1, ::8, ::8]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_xy_y.csv'), index=None,
                                                                header=None)
        pd.DataFrame(random_smoothed_field_y_z[0, :, :]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_yz_y.csv'), index=None,
                                                                header=None)
        pd.DataFrame(random_smoothed_field_y_z[1, :, :]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_yz_z.csv'), index=None,
                                                                header=None)
        pd.DataFrame(random_smoothed_field_x_z[0, :, :]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_xz_x.csv'), index=None,
                                                                header=None)
        pd.DataFrame(random_smoothed_field_x_z[1, :, :]).to_csv(os.path.join(time_export_directory,'random_smoothed_field_xz_z.csv'), index=None,
                                                                header=None)

        wrapped_mask_array = sn.map_coordinates(mask_array, np.stack(warpped_coordinates), order=0)
        geometric_structure_element = scipy.ndimage.generate_binary_structure(3, 1)
        wrapped_mask_array = scipy.ndimage.binary_opening(wrapped_mask_array, structure=geometric_structure_element)
        wrapped_mask_array = scipy.ndimage.binary_fill_holes(wrapped_mask_array)
        # wrapped_mask_array_reduced = wrapped_mask_array[
        #                              smoothing_sigma[0]:wrapped_mask_array.shape[0] - smoothing_sigma[0],
        #                              smoothing_sigma[1]:wrapped_mask_array.shape[1] - smoothing_sigma[1],
        #                              smoothing_sigma[2]:wrapped_mask_array.shape[2] - smoothing_sigma[2]]
        wrapped_mask_array = wrapped_mask_array.astype(int)
        wrapped_mask = sitk.GetImageFromArray(wrapped_mask_array)
        wrapped_mask = sitk.Cast(wrapped_mask, sitk.sitkUInt8)
        wrapped_mask.SetOrigin(mask.GetOrigin())
        wrapped_mask.SetSpacing(mask.GetSpacing())
        sitk.WriteImage(wrapped_mask, os.path.join(time_export_directory, 'randomized_mask.mha'), useCompression=True)
        draw_image_with_contours(image, [mask_array, wrapped_mask_array], time_export_directory, zoom_ratio=10)
        randomized_masks.append(wrapped_mask_array)
    return randomized_masks

def downsample_vector_field(vector_field_filepath, downsample_factor):
    vector_field = pd.read_csv(vector_field_filepath, header=None, index_col=None).values
    interpolator = scipy.interpolate.RectBivariateSpline(range(vector_field.shape[0]), range(vector_field.shape[1]), vector_field)
    downsampled_vector_field = interpolator(np.linspace(0, vector_field.shape[0]-1, int(vector_field.shape[0]/downsample_factor)),
                                            np.linspace(0, vector_field.shape[1]-1, int(vector_field.shape[1]/downsample_factor)))
    pd.DataFrame(downsampled_vector_field).to_csv(
        '_downsampled'.join(os.path.splitext(vector_field_filepath)), index=None,
        header=None)

def draw_image_with_contours(image, masks, export_directory,zoom_ratio=5):
    repetitions = len(masks)
    color_map = mc.get_cmap('rainbow')
    # masks = []
    # for mask_filepath in mask_filepaths:
    #     masks.append(sitk.ReadImage(mask_filepath))
    # image = sitk.ReadImage(image_filepath)
    # masks = align_masks(masks)
    # min_max_image_filter = sitk.MinimumMaximumImageFilter()
    # min_max_image_filter.Execute(image)
    # resampler = sitk.ResampleImageFilter()
    # resampler.SetOutputOrigin(masks[0].GetOrigin())
    # resampler.SetOutputSpacing(masks[0].GetSpacing())
    # resampler.SetSize(masks[0].GetSize())
    # resampler.SetDefaultPixelValue(min_max_image_filter.GetMinimum())
    # image = resampler.Execute(image)
    if isinstance(image, sitk.Image):
        image_array = sitk.GetArrayFromImage(image)
    else:
        image_array = image

    for d in range(image_array.ndim):

        slice_number = int(image_array.shape[d]/2)
        image_slice = image_array.take(slice_number,axis=d)

        mean = np.mean(image_slice)
        std = np.std(image_slice)
        lower_limit = mean-3*std
        higher_limit = mean+3*std
        image_slice[image_slice<lower_limit] = lower_limit
        image_slice[image_slice>higher_limit] = higher_limit
        image_slice = (image_slice-lower_limit)/(higher_limit-lower_limit)*255
        dim = (image_slice.shape[1] * zoom_ratio, image_slice.shape[0] * zoom_ratio)
        image_slice = np.array([image_slice, image_slice, image_slice]).transpose((1, 2, 0))
        image_slice = cv2.resize(image_slice, dim, interpolation=cv2.INTER_NEAREST)

        for i, mask in enumerate(masks):
            if isinstance(mask, sitk.Image):
                mask_array = sitk.GetArrayFromImage(mask)
            else:
                mask_array = mask
            mask_slice = mask_array.take(slice_number,axis=d)
            mask_slice = cv2.resize(mask_slice.astype(float), dim, interpolation=cv2.INTER_LINEAR)>0.5
            color = tuple([int(item) for item in np.array(color_map((i)/repetitions))*255][:3])
            # color = (color[0], color[1], color[2], int(color[3]*0.5))
            single_slice_combined_contour, hierarchy = cv2.findContours(mask_slice.astype('uint8'),
                                                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(image_slice, single_slice_combined_contour, -1, color, 5)
        # cv2.imshow('Contours', image_slice)
        cv2.imwrite(os.path.join(export_directory, 'randomed_contours_dim_{0}.png'.format(d)),image_slice)


def contour_randomization_validation_visualization_pipeline(image_filepath, mask_filepath, export_directory):
    image = sitk.ReadImage(image_filepath)
    mask = sitk.ReadImage(mask_filepath)
    mask = crop_mask(mask, margin_size = 5, allow_free_rotation = False, reduce_slice=True)
    mask = crop_mask(mask, margin_size=5, allow_free_rotation=False, reduce_slice=False)
    min_max_image_filter = sitk.MinimumMaximumImageFilter()
    min_max_image_filter.Execute(image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(mask.GetOrigin())
    resampler.SetOutputSpacing(mask.GetSpacing())
    resampler.SetSize(mask.GetSize())
    resampler.SetDefaultPixelValue(min_max_image_filter.GetMinimum())
    image = resampler.Execute(image)

    sitk.WriteImage(image, os.path.join(export_directory, 'image.mha'), useCompression=True)
    sitk.WriteImage(mask, os.path.join(export_directory, 'mask.mha'), useCompression=True)

    randomized_masks = draw_randomized_field(image, mask, export_directory,times=4)
    draw_image_with_contours(image, randomized_masks, export_directory, zoom_ratio=10)


if __name__ == '__main__':
    dataset_directory = '../DemoData'
    rois = ['CTV','Bladder', 'Rectum', 'LFemoral', 'RFemoral']
    export_directory = '../Plotting/contour_randomization'
    for roi in rois:
        image_filepath = os.path.join(dataset_directory, 'CT.mha')
        mask_filepath = os.path.join(dataset_directory, roi+'_mask.mha')
        roi_export_directory = os.path.join(export_directory, roi)
        if not os.path.exists(roi_export_directory):
            os.mkdir(roi_export_directory)
        contour_randomization_validation_visualization_pipeline(image_filepath, mask_filepath, roi_export_directory)