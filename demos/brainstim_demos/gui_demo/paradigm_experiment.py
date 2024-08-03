import math

from psychopy import monitors
import numpy as np
from metabci.brainstim.paradigm import (
    SSVEP,
    P300,
    MI,
    AVEP,
    SSAVEP,
    paradigm,
    pix2height,
    code_sequence_generate,
)
from metabci.brainstim.framework import Experiment
from psychopy.tools.monitorunittools import deg2pix

from metabci.brainstim.paradigm import N_Back, arithmetic_task


def __create_experiment(params):
    mon = monitors.Monitor(
        name="primary_monitor",
        width=59.6,
        distance=60,  # width 显示器尺寸cm; distance 受试者与显示器间的距离
        verbose=False,
    )
    w = __get_value(params, 'win_size_width', 1920)
    h = __get_value(params, 'win_size_height', 1080)
    mon.setSizePix([w, h])
    mon.save()
    bg_color_warm = np.array([-1, -1, -1])
    win_size = np.array([w, h])

    # record_frames = False    # note 是否记录每一帧的时间日志

    # esc/q退出开始选择界面
    ex = Experiment(
        monitor=mon,
        bg_color_warm=bg_color_warm,  # 范式选择界面背景颜色[-1~1,-1~1,-1~1]
        screen_id=0,
        win_size=win_size,  # 范式边框大小(像素表示)，默认 [1920,1080]
        is_fullscr=False,  # True全窗口,此时 win_size 参数默认屏幕分辨率
        record_frames=False,
        disable_gc=False,
        process_priority="normal",
        use_fbo=False,
    )
    win = ex.get_window()

    return mon, ex, win


def __run_experiment(experiment):
    experiment.run()


def __get_value(param_dict, param_name, default_val=None):
    try:
        result = param_dict[param_name]
        if isinstance(result, str) and result.isdigit():
            return int(result)
        return result

    except KeyError:
        return default_val


def run_n_back(params):
    """根据参数配置并运行N-Back实验"""

    # create experiment obeject
    _, ex, win = __create_experiment(params)

    fps = __get_value(params, 'fps', 60)  # 刷新率
    text_pos = (0.0, 0.0)  # 提示文本位置
    tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    symbol_height = 100  # 提示文本的高度

    n_back = N_Back(win=win)
    n_back.config_color(
        refresh_rate=fps,
        text_pos=text_pos,
        tex_color=tex_color,
        symbol_height=symbol_height,
    )
    n_back.config_response()
    n_back.config_experiment_info(
        num_trials_per_block=__get_value(params, 'num_trials_per_block', 4),
        display_result_time=__get_value(params, 'display_result_time', 1),
        stimuli_symbols=__get_value(params, 'stimuli_symbols', ['A', 'B', 'C', 'D']),
        n_back_types=__get_value(params, 'n_back_types', [2, 4])
    )

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = __get_value(params, 'display_time', 1)  # 范式刺激的时常
    rest_time = __get_value(params, 'rest_time', 1)  # 提示后的休息时长
    response_time = __get_value(params, 'response_time', 2)  # 在线反馈

    nrep = __get_value(params, 'num_blocks', 3)  # block数目 todo

    online = False  # 在线实验的标志
    port_addr = None  # 采集主机端口
    # port_addr = "COM8"  #  0xdefc    # 采集主机端口
    lsl_source_id = "meta_online_worker"  # source id
    ex.register_paradigm(
        "N-Back Workload",
        paradigm,
        VSObject=n_back,
        bg_color=bg_color,
        display_time=display_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,  # block 次数
        pdim="workload-n-back",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    __run_experiment(experiment=ex)


def run_arithmetic_task(params):
    # create experiment obeject
    _, ex, win = __create_experiment(params)

    fps = __get_value(params, 'fps', 60)  # 刷新率
    text_pos = (0.0, 0.0)  # 提示文本位置
    tex_color = 2 * np.array([179, 45, 0]) / 255 - 1  # 提示文本颜色
    symbol_height = 100  # 提示文本的高度
    task = arithmetic_task(win=win)
    task.config_color(
        refresh_rate=fps,
        text_pos=text_pos,
        tex_color=tex_color,
        symbol_height=symbol_height,
    )
    task.config_response()
    task.config_experiment_info(
        num_trials_per_block=__get_value(params, 'num_trials_per_block', 4),
        display_result_time=__get_value(params, 'display_result_time', 1)
    )

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = __get_value(params, 'display_time', 4)  # 范式刺激的时常
    rest_time = __get_value(params, 'rest_time', 2)  # 提示后的休息时长
    response_time = __get_value(params, 'response_time', 2)  # 在线反馈
    nrep = __get_value(params, 'num_blocks', 3)  # block数目

    online = False  # True                                       # 在线实验的标志
    port_addr = None  # 采集主机端口
    # port_addr = "COM8"  #  0xdefc    # 采集主机端口
    lsl_source_id = "meta_online_worker"  # source id
    ex.register_paradigm(
        "Arithmetic Task Workload",
        paradigm,
        VSObject=task,
        bg_color=bg_color,
        display_time=display_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,  # block 次数
        pdim="workload-arithmetic",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    __run_experiment(experiment=ex)


def run_ssavep(params):
    # create experiment object
    mon, ex, win = __create_experiment(params)

    w = __get_value(params, 'win_size_width', 1920)
    h = __get_value(params, 'win_size_height', 1080)
    win_size = np.array([w, h])

    n_elements, rows, columns = 20, 4, 5
    n_members = 8
    stim_length, stim_width = 150, 150
    stim_color, tex_color = [1, 1, 1], [1, 1, 1]
    fps = 240
    stim_time_member = 0.5
    stim_opacities = [1]
    freqs = np.array(
        [4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20, 4, 8, 12, 16, 20]
    )
    phases = np.zeros((n_elements, 1))
    basic_code = [[0, 1], [2, 3], [4, 5], [6, 7]]
    code_sequences = [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 0, 1, 2],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
        [3, 2, 1, 0],
    ]

    code = code_sequence_generate(basic_code, code_sequences)
    n_sequence = np.shape(code)[1]
    angles = np.zeros(n_elements)
    outter_deg = 4
    inner_deg = 1.5
    radius = deg2pix(outter_deg, mon) / win_size[1] * 0.7
    basic_ssavep = SSAVEP(win=win, n_elements=n_elements, n_members=n_members)
    basic_ssavep.config_pos(
        n_elements=n_elements,
        rows=rows,
        columns=columns,
        stim_length=stim_length,
        stim_width=stim_width,
    )
    basic_ssavep.stim_width = pix2height(win_size, basic_ssavep.stim_width)
    basic_ssavep.config_member_pos(
        win,
        radius=radius,
        angles=angles,
        outter_deg=outter_deg,
        inner_deg=inner_deg,
        tex_pix=256,
        sep_line_pix=16,
    )
    basic_ssavep.config_text(tex_color=tex_color, unit="height", symbol_height=0.03)
    basic_ssavep.config_stim(
        win,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
        member_degree=None,
        stim_color=stim_color,
        stim_opacities=stim_opacities,
    )
    # win.close()

    basic_ssavep.config_flash_array(
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
    )
    basic_ssavep.config_color(
        win,
        refresh_rate=fps,
        freqs=freqs,
        phases=phases,
        codes=code,
        stim_time_member=stim_time_member,
        stimtype="sinusoid",
        stim_color=stim_color,
        sizes=[[basic_ssavep.radius * 0.9, basic_ssavep.radius * 0.9]],
    )
    basic_ssavep.config_ring(
        win,
        sizes=[[basic_ssavep.radius * 2.15, basic_ssavep.radius * 2.15]],
        ring_colors=[2 * np.array([160, 160, 160]) / 255 - 1],
        opacities=stim_opacities,
    )
    basic_ssavep.config_target(
        win,
        sizes=[[basic_ssavep.radius * 0.2, basic_ssavep.radius * 0.2]],
        target_colors=[1, 1, 0],
        opacities=stim_opacities,
    )
    basic_ssavep.config_index(index_height=0.08, units="height")
    basic_ssavep.config_response()

    bg_color = np.array([-1, -1, -1])  # 背景颜色
    display_time = 0.5  # 范式开始1s的warm时长
    index_time = 1  # 提示时长，转移视线
    rest_time = 0.5  # 提示后的休息时长
    response_time = 1  # 在线反馈
    # port_addr = 'COM8'  #  0xdefc                                  # 采集主机端口
    port_addr = None
    nrep = 2  # block数目
    lsl_source_id = "meta_online_worker"  # None                 # source id
    online = False  # True                                       # 在线实验的标志
    ex.register_paradigm(
        "basic SSaVEP",
        paradigm,
        VSObject=basic_ssavep,
        bg_color=bg_color,
        display_time=display_time,
        index_time=index_time,
        rest_time=rest_time,
        response_time=response_time,
        port_addr=port_addr,
        nrep=nrep,
        pdim="ssavep",
        lsl_source_id=lsl_source_id,
        online=online,
    )

    __run_experiment(experiment=ex)
