import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from ortools.linear_solver import pywraplp

from gurobipy import *


def load_data():
    path = r"初赛"
    # path = r"决赛"
    product = pd.read_csv(f"{path}/产品需求.csv", dtype={"瓶颈物料号": str})
    line = pd.read_csv(f"{path}/工艺路线.csv")
    work_day = pd.read_csv(f"{path}/工作日历.csv")
    material = pd.read_csv(f"{path}/瓶颈物料.csv", dtype={"物料号": str})
    switch_time = pd.read_csv(f"{path}/切换时间.csv")

    switch_time.rename(columns={"切换时间（分钟）":"切换时间"},inplace=True)
    line.rename(columns={"每个产品单位加工时间（秒）":"每个产品单位加工时间"},inplace=True)

    # 工艺线路数据预处理
    line = line.groupby(["工艺路线类型", "工序号", "设备号"]).每个产品单位加工时间.max().reset_index()

    # 修改时间格式
    work_day["日期"] = work_day["开始时间"].str[0:10]
    work_day["开始时间"] = pd.to_datetime(work_day["开始时间"])
    work_day["结束时间"] = pd.to_datetime(work_day["结束时间"])
    work_day["秒数"] = (work_day["结束时间"] - work_day["开始时间"]).dt.total_seconds()

    total_days = list(set(work_day["日期"]))
    total_days = sorted(total_days)

    zero_time = pd.to_datetime(total_days[0])
    # 识别班次信息
    work_day['开始秒数'] = (work_day["开始时间"] - zero_time).dt.total_seconds()
    work_day['结束秒数'] = (work_day["结束时间"] - zero_time).dt.total_seconds()
    work_day['班次时段'] = list(zip(work_day['开始秒数'], work_day['结束秒数']))

    dict_device_shift = work_day.groupby(['设备号', '日期']).agg({'班次时段': list}).reset_index()
    dict_device_shift = dict(
        zip(zip(dict_device_shift['设备号'], dict_device_shift['日期']), dict_device_shift['班次时段']))

    product_total_day = sorted(list(set(product["需求日期"])))

    work_day = work_day.groupby(["设备号", "日期"]).秒数.sum().reset_index()

    start1 = datetime.strptime(total_days[0], "%Y-%m-%d")
    start2 = datetime.strptime(product_total_day[0], "%Y-%m-%d")

    end1 = datetime.strptime(total_days[-1], "%Y-%m-%d")
    end2 = datetime.strptime(product_total_day[-1], "%Y-%m-%d")

    day_start = start1 if start1 < start2 else start2
    day_end = end1 if end1 > end2 else end2
    effect_day = []

    while day_start <= day_end:
        effect_day.append(day_start)
        day_start = day_start + timedelta(days=1)

    effect_day = [datetime.strftime(each, "%Y-%m-%d") for each in effect_day]
    day_map = dict(zip(effect_day, range(len(effect_day))))

    product["需求日期"] = product["需求日期"].apply(lambda x: day_map[x])
    work_day["日期"] = work_day["日期"].apply(lambda x: day_map[x])

    work_day = work_day.set_index(["设备号", "日期"]).秒数.to_dict()

    material["供应日期"] = material["供应日期"].apply(lambda x: day_map[x])
    material = material.set_index(["物料号", "供应日期"]).供应数量.to_dict()

    dict_shift = {day: {} for day in range(len(effect_day))}
    for k, v in dict_device_shift.items():
        if day_map[k[1]] in dict_shift:
            dict_shift[day_map[k[1]]][k[0]] = sorted(v)

    return product, line, work_day, material, len(day_map), switch_time, dict_shift, zero_time


def grb_model_stage1(product_inner, line_inner, work_day_map, material_map, total_day, switch_time):
    m = Model('stage1')

    total_day = 14

    # 工单信息
    dict_wo_info = product_inner.set_index('工单号').to_dict(orient='index')
    list_wo = sorted(dict_wo_info.keys())

    dict_prod_wo = product_inner.groupby('产品号').agg({'工单号': list}).reset_index().set_index(
        '产品号').to_dict()['工单号']

    list_product = sorted(set(product_inner['产品号']))
    # 工艺路线-工序-设备的加工时间
    dict_route_oper_device_time = dict(zip(zip(line_inner['工艺路线类型'], line_inner['工序号'], line_inner['设备号']),
                                           line_inner['每个产品单位加工时间']))

    dict_order_sorted_wo = \
        product_inner.sort_values(['订单号', '工序号']).groupby('订单号').agg({'工单号': list}).reset_index().set_index(
            '订单号').to_dict()['工单号']

    list_orders = sorted(dict_order_sorted_wo.keys())

    dict_order_tail_wo = {k: v[-1] for k, v in dict_order_sorted_wo.items()}

    # 工艺路线-工序的可选择设备
    route_oper_av_device = line_inner.groupby(['工艺路线类型', '工序号']).agg({'设备号': list}).reset_index().set_index(
        ['工艺路线类型', '工序号']).to_dict()['设备号']

    wo_av_device = [
        (wo, device)
        for wo in list_wo
        for device in route_oper_av_device[dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号']]
    ]

    device_mean_switch_time = switch_time.groupby('设备号').agg({'切换时间': "mean"}).reset_index()

    device_mean_switch_time = dict(
        zip(device_mean_switch_time['设备号'], np.round(2 * 60 * device_mean_switch_time['切换时间'])))

    list_device = sorted(device_mean_switch_time.keys())

    # 工单设备时间选择变量
    wo_device_date_select = m.addVars(
        [(wo, device, date) for (wo, device) in wo_av_device for date in range(total_day)],
        vtype=GRB.BINARY, name='wo_device_date_select')

    # 产品按时交付变量
    order_deliver_in_time = m.addVars(list_orders, vtype=GRB.BINARY, name='order_deliver_in_time')

    # 产品设备当日是否生产
    product_device_date_on = m.addVars([(prod, device, date)
                                        for prod in list_product
                                        for device in list_device
                                        for date in range(total_day)],
                                       vtype=GRB.BINARY,
                                       name='product_device_date_on')
    
    buffer_time = 3600
    """
    约束：
    1 产能时间限制: 生产时间+预计切换时间(平均切换时间*切换次数)+预留时间<=产能上限
    2 瓶颈物料上限
    3 唯一选择约束
    4 工艺路径限制
    5 交付时间判定，根据末位工序
    6 生产产品类型判定
    """

    # 1
    constr_prod_time = m.addConstrs(
        (
            sum(dict_wo_info[wo]['需求量'] *
                dict_route_oper_device_time[dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号'], device]
                * wo_device_date_select[wo, device, date]
                for wo in list_wo
                if (wo, device, date) in wo_device_date_select
                )
            + device_mean_switch_time[device] * quicksum(product_device_date_on.select("*", device, date))
            + buffer_time
            <= work_day_map.get((device, date), buffer_time)
            for device in list_device
            for date in range(total_day)
        ),
        name='constr_prod_time'
    )

    # 2
    constr_material_limit = m.addConstrs(
        (
            sum(wo_device_date_select[wo, device, date] * dict_wo_info[wo]['需求量']
                for (wo, device) in wo_av_device
                if dict_wo_info[wo]['瓶颈物料号'] == material)
            <= v
            for (material, date), v in material_map.items()
            if date in range(total_day)
        ),
        name='constr_material_limit'
    )

    # 3
    constr_wo_unique_select = m.addConstrs(
        (
            quicksum(wo_device_date_select.select(wo)) == 1
            for wo in list_wo
        ),
        name='constr_wo_unique_select'
    )

    # 4
    consstr_order_sequence = m.addConstrs(
        (
            sum(date1 * wo_device_date_select[wo1, device, date1]
                for device in route_oper_av_device[dict_wo_info[wo1]['工艺路线类型'], dict_wo_info[wo1]['工序号']]
                for date1 in range(total_day)
                ) <=
            sum(date2 * wo_device_date_select[wo2, device, date2]
                for device in route_oper_av_device[dict_wo_info[wo2]['工艺路线类型'], dict_wo_info[wo2]['工序号']]
                for date2 in range(total_day))
            for order in list_orders
            for (wo1, wo2) in zip(dict_order_sorted_wo[order][:-1], dict_order_sorted_wo[order][1:])
        ),
        name='consstr_order_sequence'
    )

    # 5
    constr_deliver_in_time = m.addConstrs(
        (
            order_deliver_in_time[order] ==
            sum(wo_device_date_select[dict_order_tail_wo[order], device, date]
                for device in route_oper_av_device[
                    dict_wo_info[dict_order_tail_wo[order]]['工艺路线类型'], dict_wo_info[dict_order_tail_wo[order]][
                        '工序号']]
                for date in range(min(total_day, dict_wo_info[dict_order_tail_wo[order]]['需求日期'] + 1)))
            for order in list_orders
        ),
        name='constr_deliver_in_time'
    )

    # 6
    constr_product_recog = m.addConstrs(
        (
            product_device_date_on[prod, device, date] >=
            wo_device_date_select[wo, device, date]
            for (prod, device, date) in product_device_date_on
            for wo in dict_prod_wo[prod]
            if (wo, device, date) in wo_device_date_select
        ),
        name='constr_product_recog'
    )

    obj_deliver = quicksum(order_deliver_in_time)
    
    obj_delay_date = sum((date - dict_wo_info[dict_order_tail_wo[order]]['需求日期'])
                         * wo_device_date_select[dict_order_tail_wo[order], device, date]
                         for order in list_orders
                         for device in route_oper_av_device[
                             dict_wo_info[dict_order_tail_wo[order]]['工艺路线类型'],
                             dict_wo_info[dict_order_tail_wo[order]][
                                 '工序号']]
                         for date in range(dict_wo_info[dict_order_tail_wo[order]]['需求日期'] + 1, total_day)
                         )

    obj_product_type = quicksum(product_device_date_on)

    m.setObjective(10 * obj_deliver - obj_delay_date, GRB.MAXIMIZE)

    m.write('stage1.lp')

    m.setParam('TimeLimit', 600)
    m.setParam('MIPGap', 1e-2)
    # m.setParam("MIPFocus", 3)
    # m.setParam("NoRelHeurTime", 60)

    m.optimize()

    result = []
    for (wo, device, date) in wo_device_date_select:
        if wo_device_date_select[(wo, device, date)].x > 0.1:
            result.append([wo, date, device])
    if len(result) > 0:
        result = pd.DataFrame(result, columns=["工单号", "日期", "设备"])

        tmp = product_inner[["需求日期", "需求量", "工单号", "工序号", "产品号", "工艺路线类型", "瓶颈物料号"]]

        result = pd.merge(left=result, right=tmp, on="工单号")

        result["加工时间"] = result.apply(
            lambda x: dict_route_oper_device_time[x["工艺路线类型"], x["工序号"], x["设备"]], axis=1)
        result["总加工时间"] = result["加工时间"] * result["需求量"]

        result = result.sort_values("工单号").reset_index(drop=True)

        result.to_excel("日排产计划-复赛-1.xlsx", index=False)


def grb_model_stage2(product_inner, line_inner, work_day_map, material_map, total_day, switch_time, daily_result,
                     shift_info, zero_time):
    # 工单信息
    dict_wo_info = product_inner.set_index('工单号').to_dict(orient='index')
    list_wo = sorted(dict_wo_info.keys())

    dict_prod_wo = product_inner.groupby('产品号').agg({'工单号': list}).reset_index().set_index(
        '产品号').to_dict()['工单号']

    list_product = sorted(set(product_inner['产品号']))
    # 工艺路线-工序-设备的加工时间
    dict_route_oper_device_time = dict(zip(zip(line_inner['工艺路线类型'], line_inner['工序号'], line_inner['设备号']),
                                           line_inner['每个产品单位加工时间']))

    dict_order_sorted_wo = \
        product_inner.sort_values(['订单号', '工序号']).groupby('订单号').agg({'工单号': list}).reset_index().set_index(
            '订单号').to_dict()['工单号']

    list_orders = sorted(dict_order_sorted_wo.keys())


    # 工艺路线-工序的可选择设备
    route_oper_av_device = line_inner.groupby(['工艺路线类型', '工序号']).agg({'设备号': list}).reset_index().set_index(
        ['工艺路线类型', '工序号']).to_dict()['设备号']

    wo_av_device = [
        (wo, device)
        for wo in list_wo
        for device in route_oper_av_device[dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号']]
    ]

    dict_device_switch_time = switch_time.set_index(['设备号', '前产品', '后产品']).to_dict(orient='index')
    dict_device_switch_time = {k: 60 * v['切换时间'] for k, v in dict_device_switch_time.items()}

    list_devices = sorted(set(switch_time['设备号']))

    date_device_wo = daily_result.groupby(['日期', '设备']).agg({'工单号': list}).to_dict()['工单号']
    dict_device_wo_by_date = {day: {} for day in range(total_day)}
    for k, v in date_device_wo.items():
        dict_device_wo_by_date[k[0]][k[1]] = v

    """
    每天循环求解
    list_virtual_wo: 为各设备构造虚拟工序，加工时间为0，默认在第0天
    dict_finished_wo: 记录已完成的工单
    list_rest_wo: 记录甩单
    list_tail_prod: 当天最后一种加工产品
    """

    virtual_wo = 'vir'

    dict_order_tail_wo = {k: v[-1] for k, v in dict_order_sorted_wo.items()}

    list_rest_wo = []

    device_virtual_prod = {device: virtual_wo for device in list_devices}

    result_total = pd.DataFrame()

    for day in range(total_day):

        if not shift_info[day]:
            continue

        arrange_wo = [wo for v in dict_device_wo_by_date[day].values() for wo in v]
        full_wo = list_rest_wo + arrange_wo
        if not full_wo:
            continue

        print(f"第{day}天,工单数{len(full_wo)},其中甩单数{len(list_rest_wo)}")
        print(list_rest_wo)

        dict_order_multi_wo = {order: [wo for wo in dict_order_sorted_wo[order] if wo in full_wo] for order in
                               list_orders}
        dict_order_multi_wo = {k: v for k, v in dict_order_multi_wo.items() if len(v) >= 2}

        m = Model('stage2')

        """
        变量:
        t(wo): 工单开始加工时间
        x(device, shift ,wo): wo是否在device的shift上加工
        y(device, wo1, wo2): device上wo1和wo2是否存在相邻前后关系
        z(device, shift, wo1, wo2): device上shift,wo1和wo2是否存在跨班次的相邻前后关系
        """



        dict_device_av_wo = {device: [] for device in shift_info[day]}

        dict_device_av_wo = {k: v+dict_device_wo_by_date[day].get(k, []) for k,v in dict_device_av_wo.items()}
        # dict_device_av_wo = dict_device_wo_by_date[day].copy()
        wo_assigned_device = {wo: [device] for device, wo_list in dict_device_av_wo.items() for wo in wo_list}
        for wo in list_rest_wo:
            wo_assigned_device[wo] = []
            for device in route_oper_av_device[dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号']]:
                if device in dict_device_av_wo:
                    dict_device_av_wo[device].append(wo)
                    wo_assigned_device[wo].append(device)

        dict_device_av_wo = {k: v + [virtual_wo] for k, v in dict_device_av_wo.items() if v}

        full_wo.append(virtual_wo)

        t_wo = m.addVars(full_wo, vtype=GRB.CONTINUOUS, name='start_time_wo')
        x_device_shift_wo_on = m.addVars([(device, shift, wo) for device, wo_list in dict_device_av_wo.items()
                                          for wo in wo_list
                                          for shift in range(len(shift_info[day][device]))],
                                         vtype=GRB.BINARY,
                                         name='x_device_shift_wo_on')
        y_device_wo_next_to = m.addVars([(device, wo1, wo2)
                                         for device in dict_device_av_wo
                                         for wo1 in dict_device_av_wo[device]
                                         for wo2 in dict_device_av_wo[device]
                                         if wo1 != wo2],
                                        vtype=GRB.BINARY,
                                        name='y_device_wo_next_to'
                                        )

        z_device_wo_next_to_shift = m.addVars([(device, shift, wo1, wo2)
                                               for device in dict_device_av_wo
                                               for shift in range(1, len(shift_info[day][device]))
                                               for wo1 in dict_device_av_wo[device]
                                               for wo2 in dict_device_av_wo[device]
                                               if wo1 != wo2],
                                              vtype=GRB.BINARY,
                                              name='z_device_wo_next_to_shift'
                                              )

        """
        约束:
        1 工单开始结束时间约束
        2 工单唯一选择约束，可以不选择
        3 前后唯一工单约束:
            a. 如果工单加工，前工单唯一
            b. 如果工单加工, 后工单唯一
            c. 虚拟工单必选，放在第0班次
        4 同班次切换时间约束
        5 跨班次切换约束识别
        6 跨班次切换时间约束
        7 工艺路径约束
        8 瓶颈物料约束,有甩单时考虑
        """

        # 1 工单开始时间约束
        constr_start_time = m.addConstrs(
            (
                t_wo[wo] >= sum(shift_info[day][device][shift][0] * x_device_shift_wo_on[device, shift, wo]
                                for device in wo_assigned_device[wo]
                                for shift in range(len(shift_info[day][device]))
                                )
                for wo in full_wo
                if wo != virtual_wo
            ),
            name='constr_start_time'
        )

        constr_end_time = m.addConstrs(
            (
                t_wo[wo]
                <= sum(
                    (shift_info[day][device][shift][1] - dict_wo_info[wo]['需求量'] *
                     dict_route_oper_device_time[
                         dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号'], device]) *
                    x_device_shift_wo_on[device, shift, wo]
                    for device in wo_assigned_device[wo]
                    for shift in range(len(shift_info[day][device]))
                )
                for wo in full_wo
                if wo != virtual_wo
            ),
            name='constr_end_time'
        )

        shift_start = min(l[0][0] for l in shift_info[day].values())
        const_virtual_time = m.addConstr(
            (
                    t_wo[virtual_wo] == shift_start
            ),
            name='const_virtual_time'
        )

        # 2 工单唯一选择约束，可以不选择
        constr_unique_select = m.addConstrs(
            (
                quicksum(x_device_shift_wo_on.select("*", "*", wo)) <= 1
                for wo in full_wo
                if wo != virtual_wo
            ),
            name='constr_unique_select'
        )

        # 3 前后唯一工单约束
        constr_unique_processor = m.addConstrs(
            (
                quicksum(y_device_wo_next_to.select(device, "*", wo)) ==
                quicksum(x_device_shift_wo_on.select(device, "*", wo))
                for device in dict_device_av_wo
                for wo in dict_device_av_wo[device]
                if wo != virtual_wo
            ),
            name='constr_unique_processor'
        )

        constr_unique_successor = m.addConstrs(
            (
                quicksum(y_device_wo_next_to.select(device, wo, "*")) ==
                quicksum(x_device_shift_wo_on.select(device, "*", wo))
                for device in dict_device_av_wo
                for wo in dict_device_av_wo[device]
                if wo != virtual_wo
            ),
            name='constr_unique_successor'
        )

        constr_unique_processor_vir = m.addConstrs(
            (
                quicksum(y_device_wo_next_to.select(device, "*", virtual_wo)) == 1
                for device in dict_device_av_wo
            ),
            name='constr_unique_processor_vir'
        )

        constr_unique_successor_vir = m.addConstrs(
            (
                quicksum(y_device_wo_next_to.select(device, virtual_wo, "*")) == 1
                for device in dict_device_av_wo
            ),
            name='constr_unique_successor_vir'
        )

        constr_vir_must_select = m.addConstrs(
            (
                x_device_shift_wo_on[device, 0, virtual_wo] == 1
                for device in dict_device_av_wo
            ),
            name='constr_vir_must_select'
        )

        big_m = (day + 2) * 24 * 3600
        # 4 同班次时间间隔: 前工序开始时间+加工时长+切换时长<=后工序开始时间
        constr_switch_time_same_shift = m.addConstrs(
            (
                t_wo[wo1] + (0 if wo1 == virtual_wo else
                             (dict_wo_info[wo1]['需求量'] * dict_route_oper_device_time[
                                 dict_wo_info[wo1]['工艺路线类型'],
                                 dict_wo_info[wo1]['工序号'], device])) +
                (dict_device_switch_time.get(
                    (device, device_virtual_prod[device], dict_wo_info.get(wo2, {'产品号': 0})['产品号']),
                    0) if wo1 == virtual_wo else dict_device_switch_time.get(
                    (device, dict_wo_info[wo1]['产品号'], dict_wo_info.get(wo2, {'产品号': 0})['产品号']),
                    0))
                <=
                t_wo[wo2] + big_m * (1 - y_device_wo_next_to[device, wo1, wo2])
                for (device, wo1, wo2) in y_device_wo_next_to
                if wo2 != virtual_wo
            ),
            name='constr_switch_time_same_shift'
        )

        # 5 跨班次切换约束识别
        constr_cross_shift_recog = m.addConstrs(
            (
                z_device_wo_next_to_shift[device, shift, wo1, wo2] >=
                y_device_wo_next_to[device, wo1, wo2] + x_device_shift_wo_on[device, shift - 1, wo1]
                + x_device_shift_wo_on[device, shift, wo2] - 2
                for (device, shift, wo1, wo2) in z_device_wo_next_to_shift
            ),
            name='constr_cross_shift_recog'
        )


        # 6 跨班次切换时间约束
        constr_switch_time_cross_shift = m.addConstrs(
            (
                t_wo[wo2] >=
                shift_info[day][device][shift][0] +
                (dict_device_switch_time.get((device, dict_wo_info.get(wo1, {'产品号': 0})['产品号'],
                                              dict_wo_info.get(wo2, {'产品号': 0})['产品号']), 0)
                 if wo1 != virtual_wo
                 else dict_device_switch_time.get((device, device_virtual_prod[device],
                     dict_wo_info.get(wo2, {'产品号': 0})['产品号']), 0))
                 + big_m * (z_device_wo_next_to_shift[device, shift, wo1, wo2] - 1)
                for (device, shift, wo1, wo2) in z_device_wo_next_to_shift
                if wo2 != virtual_wo
            ),
            name='constr_switch_time_cross_shift'
        )

        # 7 工艺路径约束
        constr_route_sequence_time = m.addConstrs(
            (
                t_wo[dict_order_multi_wo[order][wo_idx]] +
                sum(dict_wo_info[dict_order_multi_wo[order][wo_idx]]['需求量']
                    * dict_route_oper_device_time[dict_wo_info[dict_order_multi_wo[order][wo_idx]]['工艺路线类型'],
                dict_wo_info[dict_order_multi_wo[order][wo_idx]]['工序号'], device]
                    for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx]])
                <= t_wo[dict_order_multi_wo[order][wo_idx + 1]] +
                big_m * (1 - sum(x_device_shift_wo_on[device, shift, dict_order_multi_wo[order][wo_idx + 1]]
                                 for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx + 1]]
                                 for shift in range(len(shift_info[day][device]))))
                # big_m *(2-sum(x_device_shift_wo_on[device, shift, dict_order_multi_wo[order][wo_idx]]
                #     for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx]]
                #     for shift in range(len(shift_info[day][device]))) - sum(x_device_shift_wo_on[device, shift, dict_order_multi_wo[order][wo_idx+1]]
                #        for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx+1]]
                #        for shift in range(len(shift_info[day][device]))))
                for order in dict_order_multi_wo
                for wo_idx in range(len(dict_order_multi_wo[order]) - 1)
            ),
            name='constr_route_sequence_time'
        )

        constr_route_sequence_on = m.addConstrs(
            (
                sum(x_device_shift_wo_on[device, shift, dict_order_multi_wo[order][wo_idx]]
                    for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx]]
                    for shift in range(len(shift_info[day][device])))
                >= sum(x_device_shift_wo_on[device, shift, dict_order_multi_wo[order][wo_idx + 1]]
                       for device in wo_assigned_device[dict_order_multi_wo[order][wo_idx + 1]]
                       for shift in range(len(shift_info[day][device])))
                for order in dict_order_multi_wo
                for wo_idx in range(len(dict_order_multi_wo[order]) - 1)
            ),
            name='constr_route_sequence_on'
        )

        # 8 瓶颈物料约束
        constr_material_limit = m.addConstrs(
            (
                sum(x_device_shift_wo_on[device, shift, wo] * dict_wo_info[wo]['需求量']
                    for wo in full_wo
                    if wo != virtual_wo
                    if dict_wo_info[wo]['瓶颈物料号'] == material
                    for device in wo_assigned_device[wo]
                    for shift in range(len(shift_info[day][device])))
                <= v
                for (material, date), v in material_map.items()
                if date == day
            ),
            name='constr_material_limit'
        )

        obj_deliver = sum(x_device_shift_wo_on[device, shift, wo] for (device, shift, wo) in x_device_shift_wo_on
                          if wo != virtual_wo)

        obj_deliver_in_time = sum(x_device_shift_wo_on[device, shift, wo] for (device, shift, wo) in x_device_shift_wo_on
                                  for order, wo_list in dict_order_tail_wo.items()
                                  if wo in wo_list
                                  if day <= dict_wo_info[dict_order_tail_wo[order]]['需求日期'])

        obj_switch_time = sum((dict_device_switch_time.get(
                    (device, device_virtual_prod[device], dict_wo_info.get(wo2, {'产品号': 0})['产品号']),
                    0) if wo1 == virtual_wo else dict_device_switch_time.get(
                    (device, dict_wo_info[wo1]['产品号'], dict_wo_info.get(wo2, {'产品号': 0})['产品号']),
                    0))/60 * y_device_wo_next_to[device, wo1, wo2] for (device, wo1, wo2) in y_device_wo_next_to)

        # m.setObjective(obj_deliver, GRB.MAXIMIZE)

        m.setObjective(10*obj_deliver_in_time + 2*obj_deliver - 0.1*obj_switch_time,GRB.MAXIMIZE)

        m.write('stage2.lp')

        # m.setParam("OutputFlag", 0)
        m.setParam('TimeLimit', 300)
        m.setParam('MIPGap', 1e-2)
        m.setParam("NoRelHeurTime", 100)
        # m.setParam("MIPFocus", 1)
        m.setParam("FeasibilityTol", 1e-3)

        m.optimize()

        if m.Status == 3:
            m.computeIIS()
            m.write('inf.ilp')

        while not m.SolCount:
            # m.setParam("NoRelHeurTime", 300)
            m.setParam("MIPFocus", 1)
            m.optimize()

        result = []

        head_wo = {v[0]: v[-1] for v in y_device_wo_next_to if y_device_wo_next_to[v].x > 0.1 if v[1] == 'vir'}

        for (device, shift, wo) in x_device_shift_wo_on:
            if wo != virtual_wo:
                if x_device_shift_wo_on[device, shift, wo].x > 0.1:
                    start_time = zero_time + timedelta(seconds=t_wo[wo].x)
                    end_time = start_time + timedelta(seconds=dict_wo_info[wo]['需求量']
                                                              * dict_route_oper_device_time[
                                                                  dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo][
                                                                      '工序号'], device])

                    if head_wo.get(device, '') == wo:
                        wo_switch_time = dict_device_switch_time.get(
                            (device, device_virtual_prod[device],
                             dict_wo_info.get(wo, {'产品号': 0})['产品号']), 0)
                    else:
                        wo_switch_time = sum(dict_device_switch_time.get(
                            (device, dict_wo_info.get(wo1, {'产品号': 0})['产品号'],
                             dict_wo_info.get(wo2, {'产品号': 0})['产品号']), 0)
                                             for (device, shift, wo1, wo2) in z_device_wo_next_to_shift
                                             if wo2 == wo
                                             if y_device_wo_next_to[device, wo1, wo2].x + x_device_shift_wo_on[
                                                 device, shift - 1, wo1].x
                                             + x_device_shift_wo_on[device, shift, wo2].x >= 2.5
                                             ) + \
                        sum(dict_device_switch_time.get(
                            (device, dict_wo_info.get(wo1, {'产品号': 0})['产品号'],
                             dict_wo_info.get(wo2, {'产品号': 0})['产品号']), 0)
                              for (device, wo1, wo2) in y_device_wo_next_to
                              if wo2 == wo
                              for shift in range(len(shift_info[day][device]))
                              if y_device_wo_next_to[device, wo1, wo2].x + x_device_shift_wo_on[
                                  device, shift, wo1].x
                              + x_device_shift_wo_on[device, shift, wo2].x >= 2.5)
                    start_time_with_switch = start_time - timedelta(seconds=wo_switch_time)
                    result.append(
                        [dict_wo_info[wo]['订单号'], wo, device, start_time_with_switch, end_time])
        if len(result) > 0:
            result = pd.DataFrame(result,
                                  columns=["订单号", "工单号", "设备号", "开始时间", "结束时间"])
            for c in ["开始时间", "结束时间"]:
                result[c] = pd.to_datetime(result[c]).dt.strftime('%Y-%m-%d %H:%M:%S')
            tmp = product_inner[["工序号", "工单号", '产品号']]

            result = pd.merge(left=result, right=tmp, on="工单号")

            # result["加工时间"] = result.apply(
            #     lambda x: dict_route_oper_device_time[x["工艺路线类型"], x["工序号"], x["设备"]], axis=1)
            # result["总加工时间"] = result["加工时间"] * result["需求量"]

            result = result.sort_values("工单号").reset_index(drop=True)

            result.to_excel(f"实时排产-{day}.xlsx", index=False)
            result_total = pd.concat((result_total, result), axis=0)

            list_rest_wo = [wo for wo in full_wo if wo != virtual_wo if wo not in result['工单号'].tolist()]

            tail_wo = {v[0]: v[1] for v in y_device_wo_next_to if y_device_wo_next_to[v].x > 0.1 if v[-1] == 'vir'}

            for k, v in tail_wo.items():
                device_virtual_prod[k] = dict_wo_info[v]['产品号']

        pass

    result_total.to_csv("日排产计划-复赛-1.csv", index=False, encoding='utf-8')


if __name__ == '__main__':
    product, line, work_day, material, day_num_map, switch_time, shift, zero_time = load_data()

    grb_model_stage1(product, line, work_day, material, day_num_map, switch_time)

    # daily_result = pd.read_excel("日排产计划-复赛-1.xlsx")

    # daily_result = daily_result[['工单号', '日期', '设备', '总加工时间']]

    # grb_model_stage2(product, line, work_day, material, day_num_map, switch_time, daily_result, shift, zero_time)
