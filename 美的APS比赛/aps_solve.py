from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ortools.linear_solver import pywraplp


def load_data(path):
    """
    read data and parser data
    """
    # for example path = r"决赛"
    product = pd.read_csv(f"{path}/产品需求.csv", dtype={"瓶颈物料号": str})
    line = pd.read_csv(f"{path}/工艺路线.csv")
    work_day = pd.read_csv(f"{path}/工作日历.csv")
    material = pd.read_csv(f"{path}/瓶颈物料.csv", dtype={"物料号": str})
    switch_time = pd.read_csv(f"{path}/切换时间.csv")

    switch_time.rename(columns={"切换时间（分钟）": "切换时间"}, inplace=True)
    line.rename(columns={"每个产品单位加工时间（秒）": "每个产品单位加工时间"}, inplace=True)

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


def scip_model_stage1(product_inner, line_inner, work_day_map, material_map, total_day, switch_time):
    """
    @params:
     - product_inner:产品需求
     - line_inner:工艺线路
     - work_day_map:工作日历
     - material_map:瓶颈物料
     - total_day:决策周期
     - switch_time:切换时间
    """
    # 工单信息
    dict_wo_info = product_inner.set_index('工单号').to_dict(orient='index')

    dict_prod_wo = product_inner.groupby('产品号').agg({'工单号': list}).reset_index().set_index(
        '产品号').to_dict()['工单号']

    dict_order_info = {v['订单号']: v for v in dict_wo_info.values()}

    list_materials = list(set(v[0] for v in material_map.keys()))
    list_product = sorted(set(product_inner['产品号']))

    dict_route_oper_device_time = dict(
        zip(zip(line_inner['工艺路线类型'], line_inner['工序号'], line_inner['设备号']),
            line_inner['每个产品单位加工时间']))

    dict_order_sorted_wo = \
        product_inner.sort_values(['订单号', '工序号']).groupby(
            '订单号').agg({'工单号': list}).reset_index().set_index(
            '订单号').to_dict()['工单号']

    list_orders = sorted(dict_order_sorted_wo.keys())

    dict_order_tail_wo = {k: v[-1] for k, v in dict_order_sorted_wo.items()}

    route_oper_av_device = \
        line_inner.groupby(['工艺路线类型', '工序号']).agg({'设备号': list}).reset_index().set_index(
            ['工艺路线类型', '工序号']).to_dict()['设备号']

    list_rest_wo = []

    result_total = pd.DataFrame()

    device_mean_switch_time = switch_time.groupby('设备号').agg({'切换时间': "mean"}).reset_index()

    device_mean_switch_time = dict(
        zip(device_mean_switch_time['设备号'], np.round(2 * 60 * device_mean_switch_time['切换时间'])))

    list_device = sorted(device_mean_switch_time.keys())

    step_day = 2
    extend_day = 2

    list_days = [(step_day * i, step_day * (i + 1) - 1) for i in range(int(total_day / step_day))]

    if list_days[-1][-1] < total_day - 1:
        list_days.append((list_days[-1][-1] + 1, total_day - 1))

    full_wo = sorted(dict_wo_info.keys()).copy()

    assigned_wo = []

    for (start_day, end_day) in list_days:

        # Create the solver
        solver = pywraplp.Solver.CreateSolver('SCIP')

        if not solver:
            return None

        list_wo = [wo for wo in full_wo
                   if start_day <= dict_wo_info[wo]['需求日期'] <= end_day + extend_day
                   ] + list_rest_wo

        list_wo = sorted(set(list_wo))

        print(
            f"第{start_day}-{end_day}天,已拍工单数{len(assigned_wo)}, 待排工单数{len(list_wo)}, 其中甩单数{len(list_rest_wo)}")

        if not list_wo:
            continue

        wo_av_device = [
            (wo, device)
            for wo in list_wo
            for device in route_oper_av_device[dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号']]
        ]

        if not wo_av_device:
            continue

        sub_orders = sorted(set(dict_wo_info[wo]['订单号'] for wo in list_wo))
        sub_products = sorted(set(dict_wo_info[wo]['产品号'] for wo in list_wo))

        sub_devices = sorted(set(i[1] for i in wo_av_device))

        # 变量定义
        # 工单设备加工日期选择
        wo_device_date_select = {}
        for wo, device in wo_av_device:
            for date in range(start_day, end_day + 1):
                wo_device_date_select[(wo, device, date)] = solver.BoolVar(
                    f'wo_device_date_select_{wo}_{device}_{date}')

        # 订单是否如期交付
        order_deliver_in_time = {}
        for order in sub_orders:
            if dict_order_tail_wo[order] in list_wo:
                order_deliver_in_time[order] = solver.IntVar(0, solver.Infinity(), f'order_deliver_in_time_{order}')
            else:
                order_deliver_in_time[order] = solver.IntVar(0, 0, f'order_deliver_in_time_{order}')

        # 设备某日期是否生产产品
        product_device_date_on = {}
        for prod in sub_products:
            for device in sub_devices:
                for date in range(start_day, end_day + 1):
                    product_device_date_on[(prod, device, date)] = solver.BoolVar(
                        f'product_device_date_on_{prod}_{device}_{date}')

        # 预留缓冲时间
        buffer_time = 3600

        # Constraints

        # 1. 设备产能时间
        for device in sub_devices:
            for date in range(start_day, end_day + 1):
                solver.Add(
                    sum(dict_wo_info[wo]['需求量'] *
                        dict_route_oper_device_time[
                            dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号'], device]
                        * wo_device_date_select[(wo, device, date)]
                        for wo in list_wo if (wo, device, date) in wo_device_date_select)
                    + device_mean_switch_time[device] * solver.Sum(
                        product_device_date_on[(prod, device, date)] for prod in sub_products)
                    + buffer_time
                    <= work_day_map.get((device, date), buffer_time),
                    name=f'time_limit_{device}_{date}'
                )

        # 2. 物料使用量上限
        for (material, date), v in material_map.items():
            if date in range(start_day, end_day + 1):
                solver.Add(
                    sum(wo_device_date_select[(wo, device, date)] * dict_wo_info[wo]['需求量']
                        for (wo, device) in wo_av_device if dict_wo_info[wo]['瓶颈物料号'] == material)
                    <= v,
                    name=f"material_limit_{material}_{date}"
                )

        # 3. 唯一设备时间约束, 允许甩单（不加工）
        for wo in list_wo:
            solver.Add(
                solver.Sum(wo_device_date_select[(wo, device, date)]
                           for device in sub_devices for date in range(start_day, end_day + 1)
                           if (wo, device) in wo_av_device) <= 1,
                name=f'unique_select_{wo}'
            )

        # 4. 订单工艺路径限制：
        # a.后工序生产时间不能早于前工序
        for order in sub_orders:
            for (wo1, wo2) in zip(dict_order_sorted_wo[order][:-1], dict_order_sorted_wo[order][1:]):
                if wo1 in list_wo and wo2 in list_wo:
                    solver.Add(
                        solver.Sum(date1 * wo_device_date_select[(wo1, device, date1)]
                                   for device in
                                   route_oper_av_device[dict_wo_info[wo1]['工艺路线类型'], dict_wo_info[wo1]['工序号']]
                                   for date1 in range(start_day, end_day + 1))
                        <=
                        solver.Sum(date2 * wo_device_date_select[(wo2, device, date2)]
                                   for device in
                                   route_oper_av_device[dict_wo_info[wo2]['工艺路线类型'], dict_wo_info[wo2]['工序号']]
                                   for date2 in range(start_day, end_day + 1)),
                        name=f"order_sequence_{order}_{wo1}_{wo2}"
                    )
        # a.前工序加工时，才能加工后工序
        for order in sub_orders:
            for (wo1, wo2) in zip(dict_order_sorted_wo[order][:-1], dict_order_sorted_wo[order][1:]):
                if wo1 in list_wo and wo2 in list_wo:
                    solver.Add(
                        solver.Sum(wo_device_date_select[(wo1, device, date1)]
                                   for device in
                                   route_oper_av_device[dict_wo_info[wo1]['工艺路线类型'], dict_wo_info[wo1]['工序号']]
                                   for date1 in range(start_day, end_day + 1))
                        >=
                        solver.Sum(wo_device_date_select[(wo2, device, date2)]
                                   for device in
                                   route_oper_av_device[dict_wo_info[wo2]['工艺路线类型'], dict_wo_info[wo2]['工序号']]
                                   for date2 in range(start_day, end_day + 1)),
                        name=f"order_prod_sequence_{order}_{wo1}_{wo2}"
                    )

        # 5. 如期交付判定
        for order in sub_orders:
            solver.Add(
                order_deliver_in_time[order] ==
                solver.Sum(wo_device_date_select[(dict_order_tail_wo[order], device, date)]
                           for device in route_oper_av_device[
                               dict_wo_info[dict_order_tail_wo[order]]['工艺路线类型'],
                               dict_wo_info[dict_order_tail_wo[order]]['工序号']]
                           for date in range(min(total_day, dict_wo_info[dict_order_tail_wo[order]]['需求日期'] + 1))
                           if (dict_order_tail_wo[order], device, date) in wo_device_date_select)
                + 0,
                name=f'deliver_in_time_{order}'
            )

        # 6. 产品是否加工识别
        for (prod, device, date) in product_device_date_on.keys():
            for wo in dict_prod_wo[prod]:
                if (wo, device, date) in wo_device_date_select:
                    solver.Add(
                        product_device_date_on[(prod, device, date)] >=
                        wo_device_date_select[(wo, device, date)],
                        name=f'prod_recog_{prod}_{device}_{date}_{wo}'
                    )

        # Objective function
        """
        优化目标：1 如期交付 2：生产效率 
        """

        # 如期交付目标项：如果当日交付，系数为10，否则系数为：当前最远日期-交付日期
        obj_deliver = solver.Sum((10 if start_day == dict_order_info[order]['需求日期']
                                  else max(0, end_day + extend_day - dict_order_info[order]['需求日期'] + 1)) *
                                 order_deliver_in_time[order]
                                 for order in sub_orders)

        # 生产量目标项：如果涉及瓶颈物料，系数为2，否则为1
        obj_production = solver.Sum(dict_wo_info[wo]['需求量'] *
                                    (2 if dict_wo_info[wo]['瓶颈物料号'] in list_materials else 1) *
                                    wo_device_date_select[wo, device, date]
                                    for (wo, device, date) in wo_device_date_select)

        solver.Maximize(obj_deliver + 1e-3 * obj_production)
        solver.EnableOutput()

        # Export the model to an LP file
        # lp_string = solver.ExportModelAsLpFormat(False)
        # with open('stage1_scip.lp', 'w') as f:
        #     f.write(lp_string)

        # Set solver parameters
        solver.SetTimeLimit(60000)  # Time limit in milliseconds
        # solver.setPresolve(scip_para_setting.FAST)

        # solver.SetSolverSpecificParametersAsString("""
        #     heuristics/feaspump/freq = 10,
        #     parallel/maxnthreads = 4
        #     presolving/maxrounds = 1000
        # """)

        solver.Solve()

        # Print the objective value

        print('Objective value:', solver.Objective().Value())

        result = []
        for (wo, device, date) in wo_device_date_select:
            if wo_device_date_select[(wo, device, date)].solution_value() > 0.1:
                result.append([wo, date, device])
        if len(result) > 0:
            result = pd.DataFrame(result, columns=["工单号", "日期", "设备"])

            tmp = product_inner[["需求日期", "需求量", "工单号", "工序号", "产品号", "工艺路线类型", "瓶颈物料号"]]

            result = pd.merge(left=result, right=tmp, on="工单号")

            result["加工时间"] = result.apply(
                lambda x: dict_route_oper_device_time[x["工艺路线类型"], x["工序号"], x["设备"]], axis=1)
            result["总加工时间"] = result["加工时间"] * result["需求量"]

            result = result.sort_values("工单号").reset_index(drop=True)

            list_rest_wo = [wo for wo in list_wo if wo not in result['工单号'].tolist()]

            assigned_wo += result['工单号'].tolist()

            full_wo = list(set(full_wo) - set(result['工单号']))

            print("排产订单数:", result.shape[0])

            # result.to_excel(f"日排产计划_{start_day}.xlsx", index=False)

            result_total = pd.concat((result_total, result), axis=0)

    # result_total.to_excel(file_name, index=False)

    return result_total


def heuristic_stage2(product_inner, line_inner, material_map, total_day, switch_time, daily_result,
                shift_info, zero_time, file_name):
    """
    Params:
        - product_inner : 产品需求
        - line_inner ： 工艺线路
        - material_map ： 瓶颈物料
        - total_day： 决策周期
        - switch_time ： 切换时间
        - daily_result：第一阶段结果
        - shift_info： 班次信息
        - zero_time：起点时刻
        - file_name：保存文件名
    """
    # 工单信息
    dict_wo_info = product_inner.set_index('工单号').to_dict(orient='index')
    list_wo = sorted(dict_wo_info.keys())

    list_materials = list(set(v[0] for v in material_map.keys()))

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
    dict_wo_device = dict(zip(daily_result['工单号'], daily_result['设备']))
    for k, v in date_device_wo.items():
        dict_device_wo_by_date[k[0]][k[1]] = v

    """
    每天循环求解
    dict_finished_wo: 记录已完成的工单
    list_rest_wo: 记录甩单
    list_tail_prod: 当天最后一种加工产品
    """

    dict_order_tail_wo = {k: v[-1] for k, v in dict_order_sorted_wo.items()}

    list_rest_wo = []

    device_tail_prod = {device: None for device in list_devices}

    result_total = pd.DataFrame()

    dict_order_time = {order: 0 for order in list_orders}

    for day in range(total_day):

        if not shift_info[day]:
            continue

        arrange_wo = [wo for v in dict_device_wo_by_date[day].values() for wo in v]
        full_wo = list_rest_wo + arrange_wo
        if not full_wo:
            continue

        print(f"第{day}天,工单数{len(full_wo)},其中甩单数{len(list_rest_wo)}")
        print(list_rest_wo)

        material_usage = {material: 0 for material in list_materials}

        dict_order_wo_list = {order: [wo for wo in dict_order_sorted_wo[order] if wo in full_wo] for order in
                              list_orders}

        # 订单当天待加工工单集合
        dict_order_wo_list = {k: v for k, v in dict_order_wo_list.items() if v}

        dict_order_wo_length = {k: len(v) for k, v in dict_order_wo_list.items()}

        # 设备状态
        device_status = {device: {'product': device_tail_prod.get(device, None),
                                  'finish_time': shift_info[day][device][0][0],
                                  'wo': [],
                                  'shift': 0}
                         for device in shift_info[day]
                         if shift_info[day][device]
                         }

        _continue = True

        result = []

        to_do_wo = full_wo.copy()

        """
        筛选逻辑：
        1. 识别订单最早工单集合
        2. 按如下标准排序：
            0. 必须当日可排: 满足摆放时间，工艺路径，瓶颈物料
            1. 是否当天交期
            2. 是否末工序
            3. 最早可开始时间，考虑切换和班次时间
            4. 工单长度
            5. 需求日期
        3. 按排序放置工单，并更新设备状态
        """

        while _continue:
            list_av_wo = {v[0] for v in dict_order_wo_list.values() if v if dict_wo_device[v[0]] in device_status}
            to_sort_wo = []
            wo_info = {}
            for wo in list_av_wo:
                device = dict_wo_device[wo]
                _switch_time = dict_device_switch_time.get(
                    (device, device_status[device]['product'], dict_wo_info[wo]['产品号']), 0)
                prod_time = dict_wo_info[wo]['需求量'] * \
                            dict_route_oper_device_time[
                                dict_wo_info[wo]['工艺路线类型'], dict_wo_info[wo]['工序号'], device]

                time_available = False

                # 校验瓶颈物料
                if dict_wo_info[wo]['瓶颈物料号'] in list_materials:
                    if material_usage[dict_wo_info[wo]['瓶颈物料号']] + dict_wo_info[wo]['需求量'] > \
                            material_map[dict_wo_info[wo]['瓶颈物料号'], day]:
                        continue

                # 设备完工时间与前工序完成时间取最大，作为当前工单的最早可用时间
                start_time = max(dict_order_time[dict_wo_info[wo]['订单号']], device_status[device]['finish_time'])

                # 按班次循环，找到最早可放置的班次
                for shift in range(device_status[device]['shift'], len(shift_info[day][device])):
                    start_time = max(start_time, shift_info[day][device][shift][0])
                    if start_time + _switch_time + prod_time <= \
                            shift_info[day][device][shift][-1]:
                        time_available = True
                        break

                if time_available:
                    to_sort_wo.append((
                        1 if dict_wo_info[wo]['需求日期'] == day else 9,
                        1 if wo == dict_order_tail_wo[dict_wo_info[wo]['订单号']] else 0,
                        start_time,
                        -dict_order_wo_length[dict_wo_info[wo]['订单号']],
                        999 if dict_wo_info[wo]['需求日期'] < day else dict_wo_info[wo]['需求日期'],
                        wo
                    ))

                    wo_info[wo] = {'start_time': start_time,
                                   'end_time': start_time + prod_time + _switch_time,
                                   'shift': shift,
                                   'product': dict_wo_info[wo]['产品号']}

            if to_sort_wo:
                to_sort_wo.sort()

                selected_wo = to_sort_wo[0][-1]

                # 更新设备状态: 产品， 班次， 结束时间
                device_status[dict_wo_device[selected_wo]]['shift'] = wo_info[selected_wo]['shift']
                device_status[dict_wo_device[selected_wo]]['product'] = dict_wo_info[selected_wo]['产品号']
                device_status[dict_wo_device[selected_wo]]['finish_time'] = wo_info[selected_wo]['end_time']

                # 更新订单当前完成时间
                dict_order_time[dict_wo_info[selected_wo]['订单号']] = wo_info[selected_wo]['end_time']

                # 更新当天瓶颈物料使用
                if dict_wo_info[selected_wo]['瓶颈物料号'] in list_materials:
                    material_usage[dict_wo_info[selected_wo]['瓶颈物料号']] += dict_wo_info[selected_wo]['需求量']

                # 计算工单开始结束时间并保存结果
                start_time = zero_time + timedelta(seconds=wo_info[selected_wo]['start_time'])
                end_time = zero_time + timedelta(seconds=wo_info[selected_wo]['end_time'])

                result.append(
                    [dict_wo_info[selected_wo]['订单号'], selected_wo, dict_wo_device[selected_wo],
                     start_time, end_time])

                # print(dict_wo_info[selected_wo]['订单号'], selected_wo, dict_wo_device[selected_wo],
                #       start_time, end_time)

                # 当前订单移除工单，当天工单移除工单
                dict_order_wo_list[dict_wo_info[selected_wo]['订单号']].remove(selected_wo)
                to_do_wo.remove(selected_wo)

                device_tail_prod[dict_wo_device[selected_wo]] = dict_wo_info[selected_wo]['产品号']

            else:
                _continue = False

        list_rest_wo = to_do_wo

        if len(result) > 0:
            result = pd.DataFrame(result,
                                  columns=["订单号", "工单号", "设备号", "开始时间", "结束时间"])
            for c in ["开始时间", "结束时间"]:
                result[c] = pd.to_datetime(result[c]).dt.strftime('%Y-%m-%d %H:%M:%S')
            tmp = product_inner[["工序号", "工单号", '产品号']]

            result = pd.merge(left=result, right=tmp, on="工单号")

            result = result.sort_values("工单号").reset_index(drop=True)

            result_total = pd.concat((result_total, result), axis=0)

    result_total.to_csv(file_name, index=False, encoding='utf-8')


if __name__ == '__main__':
    path = r'决赛'
    # 数据处理
    product, line, work_day, material, day_num_map, switch_time, shift, zero_time \
    = load_data(path)
    # 结果文件名
    file_name = '日排产计划_拆分'
    # 日排程计算
    daily_result = scip_model_stage1(product, line, work_day, material, day_num_map, \
                                     switch_time)
    # 精细排程计算
    heuristic_stage2(product, line,  material, day_num_map, switch_time, 
                     daily_result[['工单号', '日期', '设备']],
                    shift, zero_time, f"{file_name}.csv")
