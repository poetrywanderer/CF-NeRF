import os
import time
from tensorboardX import SummaryWriter

class TBVisualizer:
    def __init__(self, basedir, expname):
        self._save_path = os.path.join(basedir, expname, 'summaries')
        self._log_path = os.path.join(self._save_path, 'loss_log.txt')
        self._tb_path = os.path.join(self._save_path, 'summary.json')

        # create summary writers
        self._writer_full = SummaryWriter(self._save_path, filename_suffix="_full")

        # init log file with header
        self._init_log_file()

    def __del__(self):
        self._writer_full.close()

    def _init_log_file(self):
        with open(self._log_path, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def display_current_results(self, visuals, it, is_train, save_visuals=False):
        # add visuals to events file
        for label, image_numpy in visuals.items():
            sum_name = '{}/{}'.format('Train' if is_train else 'Val', label)

            # img: 3xHxW
            if image_numpy.ndim == 3:
                if image_numpy.shape[2] == 3 or image_numpy.shape[2] == 1:
                    image_numpy = image_numpy.transpose([2, 0, 1])
                self._writer_full.add_image(sum_name, image_numpy, it)
            else:
                self._writer_full.add_video(sum_name, image_numpy, it)

            # save image to file
            if save_visuals:
                util.save_image(image_numpy,
                                os.path.join(self._opt.checkpoints_dir, self._opt.name,
                                             'event_imgs', sum_name, '%08d.png' % it))

        # force write
        self._writer_full.file_writer.flush()

    def display_mesh(self, mesh, it):
        # add mesh to tensorboardX

        self._writer_full.add_mesh('mesh', vertices = mesh, global_step=it)

        # force write
        self._writer_full.file_writer.flush()

    def plot_scalars(self, scalars, it, is_train, is_mean=False):
        for label, scalar in scalars.items():
            # set labels
            if is_mean:
                label = f"M_{label}"
            sum_name = '{}/{}'.format('Train' if is_train else 'Val', label)

            # add scalars to events file
            self._writer_full.add_scalar(sum_name, scalar, it)

        # force write
        self._writer_full.file_writer.flush()

    def plot_histograms(self, histograms, it, is_train, is_mean=False):
        for label, scalar in histograms.items():
            # set labels
            if is_mean:
                label = f"M_{label}"
            sum_name = '{}/{}'.format('Train' if is_train else 'Val', label)

            # add hist to events file
            self._writer_full.add_histogram(sum_name, scalar, it)

        # force write
        self._writer_full.file_writer.flush()

    def plot_time(self, read_time, train_time, it):
        # add scalars to events file
        self._writer_full.add_scalar("read_time", read_time, it)
        self._writer_full.add_scalar("train_time", train_time, it)

    def print_current_train_errors(self, epoch, i, iters_per_epoch, errors, iter_read_time,
                                   iter_procs_time, visuals_were_stored):
        # set label
        log_time = time.strftime("[%d/%m %H:%M:%S]")
        visuals_info = "v" if visuals_were_stored else ""
        message = '%s (T%s, epoch: %d, it: %d/%d, s/smpl: %.3fr %.3fp) ' % (log_time, visuals_info, epoch, i,
                                                                            iters_per_epoch, iter_read_time,
                                                                            iter_procs_time)
        # print in terminal and store in log file
        self._print_and_store_errors(errors, message)

    def print_current_validate_errors(self, epoch, errors, t):
        # set label
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (V, epoch: %d, time_to_val: %ds) ' % (log_time, epoch, t)

        # print in terminal and store in log file
        self._print_and_store_errors(errors, message)

    def print_epoch_avg_errors(self, epoch, errors, is_train):
        # set label
        label = "MT" if is_train else "MV"
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s (%s, epoch: %d) ' % (log_time, label, epoch)

        # print in terminal and store in log file
        self._print_and_store_errors(errors, message)

    def _print_and_store_errors(self, errors, message):
        # set errors msg
        for k, v in errors.items():
            message += '%s:%.3f ' % (k, v)

        # print in terminal and store in log file
        print(message)
        self._save_log(message)

    def print_msg(self, message):
        # set label
        log_time = time.strftime("[%d/%m/%Y %H:%M:%S]")
        message = '%s %s' % (log_time, message)

        # print in terminal and store in log file
        print(message)
        self._save_log(message)

    def save_images(self, visuals):
        for label, image_numpy in visuals.items():
            image_name = '%s.png' % label
            save_path = os.path.join(self._save_path, "samples", image_name)
            util.save_image(image_numpy, save_path)

    def _save_log(self, msg):
        with open(self._log_path, "a") as log_file:
            log_file.write('%s\n' % msg)
