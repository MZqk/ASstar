# 深空天文后期处理中场旋、黑边与不可靠边缘像素的裁切逻辑

生成日期：2026-08-04

调研口径：不限年代，优先软件/天文机构官方文档、同行评审论文及可验证实现。所有明确标为不确定的字段和值均不进入本报告正文，原始标记保留在 `results/*.json`。

## 执行摘要

深空图像边缘裁切不应被理解为“找到黑色像素后切掉”，而应被建模为可靠支持域判定：先利用配准变换、footprint、覆盖计数、权重和数据质量掩膜判断像素是否具有足够观测支持；只有缺少这些证据时，才退化到最终图像的近黑比例、亮度台阶、色偏和连通性检测。[Siril 的注册文档](https://siril.readthedocs.io/en/stable/preprocessing/registration.html)明确提供 `current/min/max/cog` framing，并说明 `min` 是所有图像的共同区域；[reproject](https://reproject.readthedocs.io/en/stable/footprints.html)则把 footprint 作为重投影的标准输出，甚至可以表达分数覆盖。

推荐把边缘处理分成两层：`geometric_mask` 表示像素是否落入输入视场，`reliable_mask` 再叠加最低覆盖、DQ/rejection、插值核腐蚀和下游支持域。矩形交付链路从可靠掩膜中求受目标 ROI 约束的最大安全矩形；内部空洞、坏列、低覆盖岛和马赛克非矩形边界继续用 mask/weight 表达，不应为了一个内部洞牺牲大面积视场。Drizzle 的权重和噪声还受 `pixfrac`、尺度、几何与抖动方式影响，因此不存在跨数据集通用的固定像素保护带。[STScI DrizzlePac](https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise)提供了这一点的直接依据。

裁切必须作为原子数据操作处理：SCI、WCS、mask、uncertainty、weight、coverage、context 和 rejection plane 同步裁切并校验；裁后还需为测光孔径、背景环、PSF、卷积、反卷积和神经网络保留各自安全边界。[Astropy CCDData](https://docs.astropy.org/en/stable/nddata/ccddata.html)展示了 mask/flags/uncertainty 与图像数据并存且裁切时保持 WCS 更新的数据模型；[Photutils](https://photutils.readthedocs.io/en/latest/api/photutils.aperture.decode_aperture_flags.html)则显式区分 no-overlap、partial-overlap、masked、non-finite 和 too-few-pixels 等测量失败状态。

### 推荐决策逻辑

1. 在注册或叠加阶段保存每帧变换及输出 `coverage/weight/DQ/rejection`；外部母版优先读取已有扩展或旁车。
2. 构造 `geometric_mask`，再按最低覆盖、异常标志、插值支持域和下游算子半径腐蚀为 `reliable_mask`。
3. 没有辅助证据时才启动像素 fallback，并要求近黑、亮度台阶、色偏、跨尺度一致性等至少两类软证据共同支持。
4. 根据产品目的选择 `common/min`、`reference/current` 或 `union/max + mask`；多通道合成使用所需通道可靠区的交集。
5. 在原始网格上求 ROI 约束的候选矩形，先模拟面积、角视场、目标/参考星和 WCS 损失，再决定是否提交。
6. 提交后同步裁切所有辅助平面、更新并验证 WCS；内部洞继续保留为 mask。
7. 输出 `science-safe / processing-safe / display-only / degraded / manual-review` 分类、剩余风险和完整 provenance。

### 对当前项目的结论

当前 Stage 2 位于背景建模之前，阶段位置正确，也已经具备四边非对称检测、迭代复检和累计裁切报告；Siril 官方同样警告场旋、dithering 或不完整重叠形成的黑边会污染自动背景模型，要求先清理边缘。[Siril 背景提取文档](https://siril.readthedocs.io/en/latest/processing/background.html)

主要缺口是 Stage 1 没有保留逐像素 coverage/weight，Stage 2 因而把 RGB 亮度扫描同时当作几何有效性和视觉伪影判据；同时缺少统一总裁切预算、目标 ROI、失败候选回滚、辅助平面/WCS 事务和续跑旁车恢复。建议按 P0/P1/P2 推进：P0 先修预算、回滚、配置接线、非有限值状态和 provenance；P1 再从 Stage 1 引入 coverage/weight 与可靠掩膜；P2 才扩展 common/reference/union+mask 以及分级产品。具体证据和实施项见后文“当前项目 Stage 2 差距与实施建议”。

## 目录

1. [边缘缺陷分类与有效像素语义](#item-1) — 推荐处理阶段：证据生成应前置：配准/重投影时就生成 footprint，叠加时同步生成 coverage、weight、context、rejection 和 uncertainty。最终裁切推荐…；验收与回退：science-safe：保留 union 及完整 mask/weight/uncertainty，裁切不丢指定科学 ROI；processing-safe：矩形内部均满足可靠覆盖并…
2. [配准变换 Footprint 与覆盖权重检测](#item-2) — 推荐处理阶段：footprint 必须在配准变换确定时生成，coverage/context/weight/rejection 必须在叠加时累计；这是最接近事实来源、成本最低的阶段。物理裁切在最终…；验收与回退：science-safe：union 数据、mask/weight/uncertainty 与 WCS 完整，可定量解释每像素贡献；processing-safe：候选矩形内所有像素…
3. [基于最终图像统计的退化检测](#item-3) — 推荐处理阶段：首选在注册或叠加时生成直接覆盖证据；像素回退检测则应在叠加完成、仍为线性数据且背景提取、光度校色、反卷积、去星、降噪和拉伸之前运行。Siril 明确指出自动背景方法会拟合包括边缘在内…；验收与回退：科学安全: 只有原始非有限值或明确哨兵值、边界连通、裁后辅助信息同步且 ROI 完整时，可将自动结果视为较高置信的科学安全裁切。；处理安全: 两个以上独立软指标、跨尺度一致、裁后显著…
4. [裁切几何、最大可靠核心与内部空洞](#item-4) — 推荐处理阶段：最优做法是在 Stage 1 配准和叠加过程中生成并保留 mask、coverage、weight 和 context，在选定输出 WCS 时一次性确定 framing；Stage…；验收与回退：科学安全: 基于 footprint、DQ、coverage 或 weight 的可靠 mask，矩形内没有不允许的洞，满足任务覆盖与 ROI，所有辅助平面和 WCS 同步。；处理安…
5. [插值支持域与下游安全保护带](#item-5) — 推荐处理阶段：基础 footprint、coverage 和可靠掩膜应在每次配准/重采样与叠加时生成，而不是等到最终图像再猜。；用于背景提取、校色、源检测的第一版裁切/掩膜应在叠加后、背景建模前完…；验收与回退：science-safe：ROI、参考星与背景环均处于可靠掩膜内，覆盖/权重达标，所有计划算子支持完整，辅助平面和 WCS 同步。；processing-safe：可安全执行背景、校…
6. [裁切决策、目标保护与停止条件](#item-6) — 推荐处理阶段：第一版边界决策在配准叠加完成后、背景提取前执行，避免黑边/低覆盖污染背景统计；Siril 官方教程也是先裁边，再背景提取、光度校色和反卷积。；目标 ROI 和产品用途必须在最终自动接…；验收与回退：science-safe：高优先证据齐全；ROI/参考星/背景环完整；硬无效为零；coverage/weight 和算子支持达标；WCS 与辅助平面同步。；processing-sa…
7. [多通道、CFA、马赛克与产品目的](#item-7) — 推荐处理阶段：先在校准后的帧或各通道主图上完成天球对齐，选定所有联合产品共享的输出 WCS，并在配准/叠加时生成逐通道 coverage 与 mask。最终共同矩形应在线性阶段、背景建模和任何跨通…；验收与回退：科学安全: 保留每通道 SCI、WCS、mask、coverage、weight 和不确定度；单通道测量可使用各自可靠区，多通道测量明确所需通道交集，CFA 相位和来源完整。；处理安…
8. [WCS 与辅助数据传播及科学测量边界](#item-8) — 推荐处理阶段：在注册/叠加阶段生成 WCS 与辅助平面，在线性图像上完成最终几何裁切，并早于全图背景建模、源检测目录、测光校色、PSF 建模、反卷积、降噪、去星和拉伸。若产品已有可信 WCS，裁切…；验收与回退：科学安全: SCI、WCS 和全部辅助平面原子同步，可靠 mask 有直接数据来源，测量按各自支持域筛选，provenance 完整且 WCS 同点测试通过。；处理安全: 矩形及必要…
9. [主流软件与天文数据管线实践](#item-9) — 推荐处理阶段：几何画幅应在配准变换确定后选择；若能在叠加时直接输出 common/reference/ROI，可避免先生成大面积黑边再依亮度猜测。；最终可靠裁切应在叠加后、背景建模和校色前完成，并…；验收与回退：science-safe：footprint/DQ/weight/context 或 area 等高优先证据齐全，ROI 和辅助平面完整，WCS 验证通过。；processing-s…
10. [当前项目 Stage 2 差距与实施建议](#item-10) — 推荐处理阶段：保留 Stage 2 位于 Stage 1 叠加之后、Stage 3 背景处理之前的顺序；这能在最早阶段隔离黑边对背景、校色和去星的污染。；Stage 1 应同时产出 framing…；验收与回退：现状仅 ok/degraded；no-improvement、最终 edge_black 超标、裁切命令失败和保存失败都可能归为 degraded，但原因语义不同。；现状即使 deg…

<a id="item-1"></a>
## 1. 边缘缺陷分类与有效像素语义

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 建立深空图像在配准、重采样、叠加及后期处理后的像素状态模型，回答哪些边缘必须裁切、哪些只应掩膜或降低权重、哪些是合法天文信号。范围包括场旋黑角、平移或畸变变换后的空白、低覆盖、拒绝样本、插值像素、真实暗背景、负值、校准缺陷、通道错位、drizzle 空洞与卷积振铃。不把普通构图取舍、全图梯度校正、星点修复或内部孤立坏点一概转化为四边裁切问题。核心原则是“数据语义优先于显示亮度”：是否有效由变换 footprint、DQ/mask、覆盖与权重决定，像素值仅是降级证据。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 无数据：输出坐标没有任何输入像素支持，通常应由 footprint=0、coverage=0、weight=0 或显式 fill/NaN 标识；显示为黑只是表现，不是定义。
- 几何有效但数值无效：坐标落在某帧 footprint 内，但对应输入被 DQ、坏点、饱和、宇宙线、卫星轨迹或用户掩膜排除；若所有候选都被排除，则输出无可靠测量。
- 低覆盖：至少有有效贡献，但参与帧数、有效曝光或逆方差权重显著低于内部基准；它不是 no-data，却有更大噪声、更不稳定的拒绝统计和背景统计。
- 拒绝样本：某些输入值因离群判定未进入叠加；拒绝本身不等于输出像素无效，需结合剩余贡献数和拒绝比例判断。
- 插值或重投影像素：由邻域样本计算而非传感器原始采样，通常可用于常规后期，但噪声相关性、PSF 和边界支持域已经改变；必须保留 footprint/weight。
- 真实暗背景：有限、具有有效覆盖和正常权重的低信号天空、暗星云或吸收带，应保留；不能因接近黑电平而自动裁掉。
- 合法负值：偏置、暗场或背景扣除后的线性数据可以低于零，只要覆盖、DQ 和不确定度正常就仍是有效测量；负值不能单独作为 no-data 判据。
- 校准缺陷：热像素、冷像素、坏列、饱和、平场异常、放大器辉光或过扫描残留由 DQ、坏像素表和校准诊断识别；孤立或内部缺陷应掩膜、插值或重校准，只有形成连续边缘带时才适合裁切。
- 通道错位或通道覆盖不一致：RGB/窄带通道各自有效，但共同 footprint 较小，边缘会出现单色带；应先重对齐，再按共同可靠区裁切或保留通道 mask。
- 卷积、反卷积或重采样振铃：由核的负旁瓣、边界填充值或算法边界条件产生的过冲/欠冲；若遍布亮星周围则不是裁切问题，应修改核、边界或模型设置；仅局限外缘且无法掩膜时才加保护带裁切。

#### 缺陷成因 (`defect_causes`)

> 场旋来自曝光序列相对相机坐标的姿态变化，配准到同一方向后，各帧旋转矩形的非共同区域成为三角黑角或低覆盖楔形；平移、dither、畸变校正、尺度变化和马赛克布局同样改变 footprint。固定零值边缘可能是重采样的常数填充，NaN/特殊 fill 可能是软件明确写入的 no-data。低覆盖来自只有少数帧落入该位置、被 DQ/离群拒绝过多、drizzle 的 pixfrac/尺度过激或 CFA 各色采样稀疏。彩边多由多通道变换、裁切范围或权重不一致。亮/暗窄边也可能由平场、偏置、暗场、局部归一化和插值边界造成，不能与几何空白混为一谈。可区分证据依次是：变换投影与 footprint；coverage/context/weight；DQ、rejection 与 uncertainty；跨通道一致性；最后才是像素强度、连通形状和从边缘向内的台阶。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

> 证据优先级建议为：①每帧几何变换或 WCS 生成的 footprint；②DQ/坏点/饱和/宇宙线/用户 mask；③参与帧 context、coverage、有效曝光和逆方差 weight；④rejection map 与 uncertainty/variance；⑤多通道共同覆盖；⑥仅在前述辅助数据缺失时，使用 NaN、非有限值、已知 fill 值、近零连通区域、边缘亮度台阶、色偏与噪声突变。STScI 的 AstroDrizzle 将 SCI、WHT、CTX 分离，CTX 记录哪些曝光贡献，WHT 表示组合权重；Astropy CCDData 也把 data、mask、flags、uncertainty 和 WCS 分开。这些模型共同说明“像素值”和“像素有效性”不可合并。

#### 所需输入 (`required_inputs`)

> 最低输入为线性叠加图、原始尺寸、通道信息和裁切前坐标系；推荐增加每帧到输出的几何变换或 WCS、每帧有效 mask/DQ、coverage/context、有效曝光、逆方差 weight、rejection map、uncertainty、叠加方法及重采样核。OSC 未去马赛克数据还需 CFA 类型、ROWORDER、XBAYROFF/YBAYROFF 或等价相位元数据。

#### 算法步骤 (`algorithm_steps`)

> 一、从变换/WCS和有效输入 mask 生成每帧 footprint；二、聚合 coverage、有效曝光、weight、context 和 rejection；三、构造几何 mask（至少一个输入落入）与可靠 mask（覆盖、权重、DQ、拒绝和通道条件均满足）；四、对可靠 mask 做与重采样核支持域相匹配的腐蚀；五、标记与画布外边界连通的不可靠区域，内部洞单独处理；六、根据用途求最大轴对齐内接矩形、共同覆盖矩形或保留全画幅+mask；七、检查目标 ROI、弱信号和参考星是否被切断，计算面积和角视场损失；八、同步裁切 SCI、mask、weight、uncertainty、context/rejection 并更新 WCS；九、复检四边可靠度、通道一致性、背景统计和 WCS 往返误差；十、若没有辅助平面，才运行基于外边界连通性、亮度/噪声台阶和色偏的低置信度检测，并输出人工复核。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 几何有效区 G 只回答输出坐标是否落入至少一个输入 footprint；可靠区 R 还要求输入未被 DQ/mask 排除，并满足最小参与帧数、有效曝光、权重、拒绝比例、不确定度与所需通道共同覆盖。G 内可以存在低覆盖、零权重或全部被拒绝的像素；R 通常是 G 的子集。显示用画幅可以采用 G 或 union，背景建模、校色、反卷积、测光和训练/推理类算法应使用更严格的 R 或显式 mask。

#### 内部空洞策略 (`internal_hole_policy`)

> 内部无效岛、坏列、芯片间隙和 drizzle holes 不应通过四边裁切解决，否则可能牺牲大量有效视场。小而稀疏的洞使用 mask 并由能识别 mask 的算法跳过，显示副本可作受控插值；长坏列、芯片间隙或密集 drizzle 空洞保留区域 mask，必要时调整 dither、pixfrac、scale 或重做叠加。只有内部洞把候选矩形切成不可用区域，且下游完全不支持 mask 时，才选择避开洞的最大内接矩形或降级为人工构图。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> 证据生成应前置：配准/重投影时就生成 footprint，叠加时同步生成 coverage、weight、context、rejection 和 uncertainty。最终裁切推荐在叠加完成后立即决定，并位于背景提取、测光/光度颜色校准、反卷积、去星、降噪和非线性拉伸之前，因为不可靠边缘会污染这些步骤的统计或边界条件。科学数据产品可不物理裁切，而是保留 union 画幅和完整 mask；面向不支持 mask 的后续流水线则输出 processing-safe 裁切副本。Siril 官方 FAQ 同样把叠加后的裁切列为背景去除、颜色校准和反卷积之前的第一步。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

> 按目的区分三种画幅：science-union 保留所有几何有效像素并携带 mask/weight；common-coverage 取所有指定输入或通道的交集，最稳健但损失最大；processing-safe 在可靠 mask 内求最大轴对齐矩形，允许四边非对称裁切。场旋黑角适合可靠 mask 的内接矩形，简单四边扫描只是其近似；大角度旋转或马赛克的凹形覆盖不能仅用每边黑像素比例可靠求解。展示构图可在 processing-safe 区内再裁，但不得覆盖科学裁切 provenance。Siril registration 提供 current、max/外接框、min/共同区和中心重心等 framing，说明 union 与 common-area 本来就是不同政策。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 裁切前应在 WCS 或像素坐标中定义硬保护 ROI：目标主体、潮汐尾/彗尾/星云外晕等低表面亮度结构、校色和测光参考星、背景采样区及计划拼接的重叠带。候选矩形若切入硬 ROI 则不得自动执行；可转为 union+mask、降低可靠度门槛、只裁某些边或人工选择。暗星云不能由亮度阈值识别为废边，弱信号保护应来自目录/WCS、用户 ROI、线性高动态范围预览和多尺度结构检查。多通道项目必须在共同 WCS 上验证每个通道 ROI 的覆盖。

#### 验收与回退 (`acceptance_and_fallback`)

> science-safe：保留 union 及完整 mask/weight/uncertainty，裁切不丢指定科学 ROI；processing-safe：矩形内部均满足可靠覆盖并适合不识别 mask 的后续算法；display-only：仅为构图或隐藏边缘，不作为测光/定量产品；degraded：只有像素统计、辅助图缺失、阈值未达标或损失预算超限；manual-review：ROI 冲突、马赛克/凹形 footprint、通道分歧或大幅裁切。回退顺序为保留全图+mask、改用较宽画幅并让下游读取 mask、重做配准/叠加、调整 drizzle/dither/拒绝参数，最后才是基于亮度的保守裁切。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

> 常见误判包括暗星云、低表面亮度外晕、正常负背景、极窄带近零天空、强渐晕、平场不足、月光/光污染梯度、马赛克的有意非矩形边界、稀疏 CFA drizzle 和通道色阶差异。亮度行列统计还会被靠边亮星、星芒、卫星轨迹、放大器辉光和大尺度星云扭曲；仅看角点会漏掉边中部缺口，仅看四边平均会掩盖窄楔或内部洞。过严 common-area 会因单帧异常位移牺牲大面积，过宽 union 又会把低覆盖噪声送入背景和反卷积。拒绝图中的天体结构可能是配准或局部归一化错误，而非真正异常；STScI 明确建议将 DQ 与科学图闪烁检查，避免把星点误标成宇宙线。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

> 未去马赛克 CFA 裁切必须保持相位：Bayer 选择框对齐 2×2 周期，X-Trans 对齐 6×6 周期，或正确更新 CFA offset；任意角旋转/缩放应在 debayer 或 Bayer drizzle 后进行。Siril 官方界面会将 Bayer 裁切吸附到 2×2、X-Trans 吸附到 6×6。已分离 RGB/窄带通道应先统一 WCS/几何变换，再以各通道可靠 mask 的交集确定供颜色校准和合成使用的范围；单通道缺失不能用黑色填充冒充有效颜色。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

> 裁切必须对 SCI、mask/DQ、uncertainty/variance、weight、coverage/context、rejection、曝光图和分割/星表使用完全相同的像素切片。FITS WCS 中需更新 NAXIS/图像尺寸和参考像素；纯平移裁切的等价操作是把 CRPIX1、CRPIX2 分别减去左、下或相应数组坐标偏移，但存在 ROWORDER、轴顺序、SIP/畸变和软件坐标约定时应使用经过验证的 WCS slicing 库，而非手改两个关键字。Astropy 的 WCS/NDData slicing 会同步切片 WCS、mask 与 uncertainty。裁后应做像素→天球→像素往返和角点 footprint 对比，并保存原 WCS 与裁切变换。

#### 下游影响 (`downstream_impact`)

> 不可靠边缘会拉低全局/边缘分位数并把背景模型拉向填充值，改变星点检测阈值、噪声估计、颜色通道比例和光度标定；反卷积、卷积降噪、去星和神经网络可能把黑边传播成振铃、亮带或假结构；非线性拉伸会放大低覆盖噪声和彩边；源检测和测光若忽略 weight/uncertainty 会低估误差或产生假源。相反，误裁真实暗背景、弱星云或参考星会造成不可逆信息损失。最稳妥的工程模式是让统计/科学算法读取 mask 与权重，并仅为不支持这些辅助面的步骤生成 processing-safe 矩形。

#### 质量验证 (`qa_and_validation`)

> 自动验收至少比较裁前后：四边/角落的 coverage、weight、有效曝光、拒绝比例、非有限或 fill 连通分量、背景中位数/MAD、通道色偏和噪声；绘制 SCI 叠加可靠 mask、coverage 热图、WHT、CTX/rejection 和候选矩形。确认候选矩形内部不存在外边界连通无效区，保护带后最小可靠度达标，ROI 完整且面积损失在预算内。WCS 验证使用原图与裁图共同天球点的往返误差、裁后四角天空坐标和源表交叉匹配。视觉验收必须在线性拉伸和强拉伸下进行，且抽查靠边星点/弱结构未被错误裁掉。

#### 裁切溯源 (`crop_provenance`)

> 记录输入文件哈希和软件版本、原始/最终尺寸、四边偏移、候选与采用矩形、画幅模式、每个 mask 定义、阈值与内部基准、腐蚀/保护带、面积和角视场损失、ROI 约束、WCS 前后摘要、辅助平面名称、证据置信度、degraded 原因、人工覆盖者/时间以及可逆恢复所需的原始产品路径。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

> Siril 官方 registration 支持 max 外接画幅与 min 共同区域；官方 FAQ 明确指出叠加彩边会污染多种图像统计，推荐先裁切再做背景、颜色校准和反卷积。Siril 1.5 的 drizzle 文档说明零输入像素会形成黑色空洞，pixfrac、scale、kernel 和帧数决定覆盖，且 CFA 红蓝通道更稀疏。STScI AstroDrizzle 的正式数据模型输出 SCI、WHT、CTX，并把 DQ、静态坏点和宇宙线 mask 合入最终权重；这比从 SCI 灰度猜边更可靠。Astropy CCDData/NDData 将数据、mask、flags、uncertainty 和 WCS 共同传播，reproject 返回独立 footprint，插值模式为 0/1 而 exact 模式可给分数覆盖。FITS WCS 论文及 Astropy slicing 支持裁切后保持坐标一致。由这些事实推断：消费级深空流水线若能在配准/叠加时保留 footprint/coverage，Stage2 应以语义 mask 为主、亮度检测为降级。

#### 直接证据来源 (`evidence_sources`)

- Siril 官方文档《Registration》，说明 current/max/min/cog framing、min 为共同区及重采样方法，访问日期 2026-08-04：https://siril.readthedocs.io/en/stable/preprocessing/registration.html
- Siril 官方 FAQ《In which order should image processing functions be used?》，说明彩色叠加边缘会影响统计且裁切应在背景/校色/反卷积前，访问日期 2026-08-04：https://siril.org/faq/
- Siril 官方文档《Drizzle》，说明 null pixels、帧数/pixfrac/scale/kernel 与 CFA 覆盖，访问日期 2026-08-04：https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- Siril 官方文档《Main interface》，说明 Bayer 2×2 与 X-Trans 6×6 裁切相位约束，访问日期 2026-08-04：https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- STScI HST 用户文档《Weight Maps and Correlated Noise》，说明 Drizzle 权重图、缺失输入和相关噪声，访问日期 2026-08-04：https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise
- STScI DrizzlePac 文档《AstroDrizzle Primary User Interface》，说明 SCI/WHT/CTX、DQ 和 inverse-variance weighting，访问日期 2026-08-04：https://drizzlepac.readthedocs.io/en/deployment/astrodrizzle.html
- STScI ACS 数据手册《calacs Processing Steps》，说明 DQ 位掩膜、坏点/饱和/宇宙线及被拒绝像素处理，访问日期 2026-08-04：https://hst-docs.stsci.edu/acsdhb/chapter-3-acs-calibration-pipeline/3-4-calacs-processing-steps
- Astropy 官方文档《CCDData Class》，说明 data、mask、flags、uncertainty 与 WCS 的独立语义及裁切更新，访问日期 2026-08-04：https://docs.astropy.org/en/stable/nddata/ccddata.html
- Astropy reproject 官方文档《Footprint arrays》，说明插值 footprint 为 0/1、exact footprint 可表达分数重叠，访问日期 2026-08-04：https://reproject.readthedocs.io/en/latest/footprints.html
- Greisen 与 Calabretta《Representations of World Coordinates in FITS》，A&A 395, 1061–1075，2002，FITS WCS 基础：https://arxiv.org/abs/astro-ph/0207407

#### 当前项目适配情况 (`current_project_fit`)

> 当前 Stage2 读取叠加后 RGB 像素，以中心暗背景估计近黑阈值，对行列的近黑比例、中位亮度、色偏和滚动亮度台阶做边缘扫描，再检查四角黑像素，追加固定 3 像素保护带，并以 edge_black_ratio 迭代补裁及额外彩边裁切。这能覆盖“没有 footprint/weight 时的显示强度降级路径”，也能处理常见场旋黑角和彩色窄边；但它无法可靠区分真实暗背景、合法负值、低覆盖与 no-data，也看不到 DQ/rejection、有效曝光、内部 drizzle holes、通道独立 footprint 或插值核支持域。其四边扫描近似轴对齐内接矩形，复杂凹形/马赛克覆盖和目标 ROI 不受保护。当前报告记录尺寸与累计裁切较好，但还缺证据层、置信度、WCS/辅助图传播和科学/展示模式。


<a id="item-2"></a>
## 2. 配准变换 Footprint 与覆盖权重检测

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 研究如何利用每帧几何变换或 WCS、输入 DQ/mask、参与帧数、有效曝光、weight/context/rejection 构造输出图的“几何有效区”和“数值可靠区”，并据此裁掉场旋、平移、畸变重投影造成的黑角或不可靠边缘。目标是形成与像素亮度无关、可复现且能传播到后续阶段的主检测路径。范围覆盖普通配准叠加、drizzle、OSC/CFA 与多通道共同画幅；不把孤立内部坏点和马赛克非矩形科学覆盖强制裁成小矩形。

#### 像素状态分类 (`pixel_state_taxonomy`)

> 对每个输出像素区分：画布外或无任何输入 footprint 的 no-data；至少一帧几何落入但所有贡献被 DQ/rejection 排除的无有效样本；有贡献但 N、有效曝光或逆方差权重偏低的低覆盖；满足可靠门槛的正常像素；由重采样核部分支持的边界像素；drizzle 零 droplet 的内部空洞；多通道中只被部分通道覆盖的颜色不完整像素；被拒绝但仍有足够其他样本的有效叠加像素。SCI 中的零、负数或 NaN 只是存储表现，最终语义由这些辅助平面决定。

#### 缺陷成因 (`defect_causes`)

> 每帧相对参考坐标的平移、旋转（包括场旋）、尺度/剪切、镜头或望远镜畸变校正会把原始矩形变成输出网格上的多边形，union 外部为 no-data、交集外部则为非共同覆盖。dither 和丢帧改变参与帧数；DQ/坏点/宇宙线/卫星轨迹/饱和与离群拒绝减少有效贡献；曝光时间、透明度、平场和读噪差异改变权重；drizzle 的 scale、pixfrac、kernel 与采样相位可产生零覆盖洞和相关噪声；RGB/窄带或 CFA 各色采样的 footprint/coverage 不一致产生彩边。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

> 首选证据链为：每帧输入边界和配准变换/WCS计算的纯几何 footprint F；输入有限性与 DQ/静态坏点/用户 mask 形成的有效输入 mask M；叠加过程中实际贡献的 context/coverage C；按曝光或逆方差聚合的有效曝光 E 与 weight W；离群拒绝计数 Rj；输出 uncertainty/variance；各通道可靠 mask 的交集。reproject 官方接口直接返回 footprint，插值重投影为 0/1，exact 球面多边形交叠可给分数覆盖。AstroDrizzle 的 CTX 用位掩膜记录贡献曝光，WHT 保存权重，SCI 与有效性信息分离。只有无法获取这些证据时才回退到 SCI 的边缘近黑、色偏和亮度台阶。

#### 所需输入 (`required_inputs`)

> 每帧图像尺寸、到共同输出网格的仿射/单应/畸变映射或完整 WCS，参考输出 WCS/shape，曝光时间与叠加权重；推荐输入每帧 DQ/坏点/饱和/用户 mask、variance/逆方差、叠加 rejection 结果、通道或 CFA 标识、重采样核及 pixfrac/scale。还需目标 ROI、用途模式和可接受视场损失预算。

#### 算法步骤 (`algorithm_steps`)

- 统一输出 WCS/shape，并保存每帧从输出像素到输入像素的逆映射；变换必须与实际叠加完全一致。
- 用全 1 的输入平面或输入像素多边形投影生成纯几何 F_i；插值模式可用二值 footprint，要求边界精度或通量守恒时用分数重叠。
- 把 finite、DQ good、坏点、饱和、轨迹和用户 mask 合成 M_i。为避免把坏点值插入邻域，应在重采样前应用 M_i，并投影有效权重而不是事后只看 SCI。
- 同步投影 1、t_i、w_i 和贡献标识，聚合得到 N、C、E、W 与 context；叠加离群判定完成后记录 accepted count、rejected count 和 rejection fraction，并更新实际有效权重。
- 分别构造 geometric-any（G>0）、geometric-common（所有指定帧满足覆盖）和 reliable（覆盖、权重、拒绝、uncertainty、通道条件均达标）mask。
- 按实际重采样核或 drizzle droplet 支持域腐蚀 reliable mask，并处理坐标取整余量；只把与画布外边界连通的不可靠区域视为裁边候选，内部洞另行保留 mask。
- 依据输出模式求矩形：共同区、可靠 mask 内最大轴对齐矩形、四边约束矩形，或保留 union+mask；加入目标 ROI 和面积/角视场损失约束。
- 将同一矩形同步应用到 SCI、DQ/mask、coverage/context、E、W、rejection、uncertainty 和所有通道；用 WCS slicing 更新坐标并写 provenance。
- 复检候选矩形内部最小/分位覆盖、边缘权重、通道交集、内部洞、目标 ROI、面积损失和 WCS 往返误差；不满足则降级为全图+mask或人工处理。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 几何 mask 只由坐标变换决定：any/union 表示至少一帧覆盖，common/intersection 表示所有选定帧均覆盖；它不考虑坏点、饱和、拒绝、曝光或噪声。可靠 mask 由实际可用样本决定，是几何 mask 与投影后的 DQ/mask、coverage、有效曝光、weight、rejection、uncertainty 及通道约束的组合。一个像素可以在所有几何 footprint 内，却因某位置跨帧都是坏点或被拒绝而不可靠；也可以只被较少帧覆盖但仍适合展示。裁切策略必须明确使用哪一层，不能把 union/common/reliable 混为“有效”。

#### 内部空洞策略 (`internal_hole_policy`)

> 将与画布边界连通的不可靠分量和内部封闭分量分开。小型坏点洞由 mask 与权重处理，显示副本可插值；drizzle 密集空洞优先增加帧数、改善 dither、提高 pixfrac、降低 scale 或更换 kernel；芯片间隙和马赛克洞保留为区域 mask。求矩形时可要求内部洞面积占比低于用途门槛，或使用严格最大内接矩形完全避开洞。若严格避洞造成巨大视场损失，应保留非矩形科学产品并将不支持 mask 的后续步骤标记为不可安全运行。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> footprint 必须在配准变换确定时生成，coverage/context/weight/rejection 必须在叠加时累计；这是最接近事实来源、成本最低的阶段。物理裁切在最终叠加图形成后、背景建模和颜色校准之前执行，以便利用实际拒绝和权重结果。若后续科学步骤能读取 mask/weight，可保留 union 产品，仅给不支持 mask 的背景、反卷积、去星或神经网络阶段提供 processing-safe 派生裁图。多通道合成前应先在共同 WCS 上计算交集，而不是每个通道独立按亮度裁切。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

> 提供至少三种模式：union/science 保留 G>0 的外接画幅并携带非矩形 mask；common 取选定帧或通道 footprint 的共同区；reliable-rectangle 在腐蚀后的可靠 mask 内求满足 ROI 与损失约束的最大轴对齐矩形。仅允许四边裁切的软件可从可靠 mask 求 top/bottom/left/right，但场旋三角角落必须以整个矩形内部无边界连通坏区为约束，不能只看四条边的平均。若允许旋转构图，可搜索带角度的最大内接矩形，但会触发再次重采样和更复杂的 WCS/CFA 约束，默认应避免。Siril 的 max/min framing 可分别作为 union/common 的官方实践参照。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 在求矩形前，将 WCS 目标区域、低表面亮度外延、彗尾/潮汐尾、校色与测光参考星、背景采样区、计划马赛克重叠带栅格化为 hard/soft ROI。hard ROI 必须被候选画幅完整包含且达到最低 coverage/weight；soft ROI 可进入多目标优化，用较少不可靠像素换取结构保留。若目标跨入低覆盖区，应输出覆盖/不确定度警告而不是静默裁掉；可保留 union+mask、降低展示门槛或建议补拍。

#### 验收与回退 (`acceptance_and_fallback`)

> science-safe：union 数据、mask/weight/uncertainty 与 WCS 完整，可定量解释每像素贡献；processing-safe：候选矩形内所有像素达到配置可靠门槛并通过核腐蚀，适合不支持 mask 的算法；display-only：允许更低覆盖和插值修补，仅供展示；degraded：缺变换、coverage 或 DQ，只能用 SCI 统计估计；manual-review：ROI 冲突、马赛克凹区、阈值敏感或损失超限。回退路径为保留 union+mask、改用支持 mask 的下游、放宽但显式标记低覆盖、重做叠加/调整 drizzle，最后才基于亮度保守裁边。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

> 变换与实际重采样器不一致会让理论 footprint 偏移；只投影角点会漏掉强畸变边界弯曲，应沿边加密采样或直接重投影 mask。把 mask 用高阶插值会产生非物理中间值，应对 mask 使用最近邻或投影权重语义；反之把所有分数覆盖简单二值化又会丢失边界质量信息。单个离群位移帧会使严格 common 区过小，需按选定帧集合或覆盖分位策略处理。WHT 可能代表曝光、逆方差或其他相对权重，不能在不知道定义时比较绝对值。rejection 高可能是宇宙线，也可能是配准/归一化错误和真实天体结构。Mosaic union 的低覆盖边缘可能是有意科学区域，不能自动删。CFA drizzle 的各色 coverage 不同，合并总 coverage 会隐藏单色缺口。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

> 未去马赛克 CFA 需为 R/G1/G2/B 或 X-Trans 相位分别累计 coverage，可靠颜色区取所需颜色平面的交集；Bayer 裁切偏移与宽高应保持 2 像素相位，X-Trans 保持 6 像素相位，或正确更新 offset。Siril 文档指出 CFA drizzle 的红蓝采样更稀疏、scale>1 需要更多帧。已 debayer 的 RGB 也应保存每通道 mask/weight；多滤镜合成必须在共同 WCS 下取交集，不能用零填充缺失通道后再依赖 RGB 像素统计。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

> 输出矩形必须一次性切片 SCI、DQ/mask、N/C/E/W、CTX、rejection、uncertainty/variance、通道权重、PSF/分割图及与像素坐标关联的目录。使用 WCS-aware slicing 更新图像尺寸与参考像素，保留 SIP/畸变和备用 WCS；纯整数左/下裁切可理解为相应 CRPIX 减去偏移，但实现应交给 Astropy WCS/NDData slicing 之类库并通过往返测试。CTX 位平面不能转为普通浮点插值；重采样或旋转时必须按其位掩膜语义重新投影/聚合。原 WCS、输出 WCS、变换链与数组轴/ROWORDER 约定写入 provenance。

#### 下游影响 (`downstream_impact`)

> coverage/weight 驱动的裁切能避免无数据和低覆盖像素污染背景面拟合、噪声/MAD、源检测阈值、光度和颜色标定；可防止反卷积、卷积降噪、去星和模型推理从黑边制造振铃或假结构。WHT/uncertainty 还能让保留的低覆盖科学区域在测光时得到正确误差，而非被粗暴丢弃。若只按 SCI 近黑裁切，会误删暗星云与合法负值，也可能漏掉被常数背景填充、数值不黑但 coverage=0 的边。若不同通道不取共同可靠区，后续颜色校准和拉伸会放大彩边。

#### 质量验证 (`qa_and_validation`)

> 生成诊断套件：每帧 footprint 轮廓、N/C/E/W 热图、CTX 贡献帧抽样、DQ 与 rejection 叠图、通道共同区、腐蚀前后 reliable mask、候选矩形和 ROI。数值验收检查 candidate 内 coverage/weight 的最小值与低分位、no-data 数、内部洞面积、拒绝比例、四边到不可靠区距离、面积/角视场损失。用少量输出坐标逆变换回各输入，核验 context 所称贡献帧确实有合法输入；比较聚合 E/W 与叠加器日志。WCS 验收做裁前裁后共同源的像素↔天空往返、四角坐标和源表匹配。建立合成旋转/平移/畸变、真实 Seestar 场旋、暗星云、极窄带、CFA drizzle 和马赛克回归样本。

#### 裁切溯源 (`crop_provenance`)

> 记录输入帧清单/哈希、选帧条件、变换和 WCS 版本、输出网格、DQ bit 选择、mask 组合表达式、weight 的物理定义、N/E/W/reference 统计、可靠阈值、kernel/pixfrac/scale、腐蚀和取整规则、所有候选矩形及选择原因、ROI 与损失指标、裁切偏移、辅助 HDU 映射、验收结果、置信度、degraded/manual override 及原始 union 产品位置。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

> Astropy reproject 官方实现可从输入 WCS 到输出 WCS返回 footprint；插值 footprint 为 0/1，exact 方式利用球面多边形交叠返回分数覆盖，mosaicking 的 coadd footprint 表示输入图贡献数量。STScI AstroDrizzle 以 WHT 组合输入权重，以 CTX 位掩膜记录哪些图像贡献，并将 DQ、静态 mask 与宇宙线 mask 应用于最终 drizzle；其文档还说明缺失数据在有足够 dither 覆盖时可被补齐、weight 可用于估计相关噪声。Siril registration 的 min 取共同区、max 取外接画幅，drizzle 文档明确 point/small pixfrac 容易产生 null pixels。Astropy CCDData/NDData 和 WCS slicing 展示了 data、mask、uncertainty 与 WCS 的同步传播。上述是一手文档事实；“可靠 mask + 核腐蚀 + 受 ROI 约束的最大矩形”是将这些成熟语义组合为本项目自动裁边的工程推导。

#### 直接证据来源 (`evidence_sources`)

- Astropy reproject 官方文档《Footprint arrays》，说明插值与 exact footprint 的覆盖语义，访问日期 2026-08-04：https://reproject.readthedocs.io/en/latest/footprints.html
- Astropy reproject 官方 API《reproject_interp》，说明输出 WCS、插值阶次、roundtrip_coords 与 output_footprint，访问日期 2026-08-04：https://reproject.readthedocs.io/en/latest/api/reproject.reproject_interp.html
- Astropy reproject 官方文档《Combining images into mosaics》，说明 coadd footprint 表示输入图贡献数量，访问日期 2026-08-04：https://reproject.readthedocs.io/en/stable/mosaicking.html
- STScI HST 用户文档《Weight Maps and Correlated Noise》，说明 Drizzle 权重累积、缺失数据和相关噪声，访问日期 2026-08-04：https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-3-weight-maps-and-correlated-noise
- STScI DrizzlePac 文档《AstroDrizzle Primary User Interface》，说明 SCI/WHT/CTX、context 位掩膜、inverse variance 与 kernel，访问日期 2026-08-04：https://drizzlepac.readthedocs.io/en/deployment/astrodrizzle.html
- STScI ACS 数据手册《calacs Processing Steps》，说明 DQ bit、坏点/饱和/宇宙线、有效 mask 和拒绝后有效曝光，访问日期 2026-08-04：https://hst-docs.stsci.edu/acsdhb/chapter-3-acs-calibration-pipeline/3-4-calacs-processing-steps
- Siril 官方文档《Registration》，说明 min/common、max/bounding box 和实际变换应用，访问日期 2026-08-04：https://siril.readthedocs.io/en/stable/preprocessing/registration.html
- Siril 官方文档《Drizzle》，说明 droplet 覆盖、null pixels、kernel、pixfrac/scale 与 CFA 稀疏覆盖，访问日期 2026-08-04：https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- Astropy 官方文档《Shared Python Interface for WCS》，说明 WCS 子区域 slicing，访问日期 2026-08-04：https://docs.astropy.org/en/stable/wcs/wcsapi.html
- Greisen 与 Calabretta《Representations of World Coordinates in FITS》，A&A 395, 1061–1075，2002：https://arxiv.org/abs/astro-ph/0207407

#### 当前项目适配情况 (`current_project_fit`)

> 当前 Stage2 只接收最终 SCI/RGB 像素和尺寸，没有 Stage1 的逐帧变换、valid mask、coverage、effective exposure、weight、context 或 rejection。它从中心背景生成黑阈值，按行列近黑比例、中位数、色偏、亮度台阶和角点扫描求四边，再以 edge_black_ratio 迭代，适合作为缺元数据时的低成本后验检测。主要缺口是：无法判断近黑到底是 no-data 还是暗天空，无法识别数值被填为背景但 coverage=0 的边，不能按参与帧数/权重区分低覆盖，也不能正确处理内部 drizzle holes 和多通道共同区；固定 3 像素 guard 与实际核无关联。Stage2 crop_report 已提供累计四边裁切和尺寸，可扩展为接收/记录 reliability evidence，而不必先重写整个 Stage2。


<a id="item-3"></a>
## 3. 基于最终图像统计的退化检测

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 研究在配准或叠加过程没有保留覆盖图、权重图、拒绝图和有效像素掩膜时，如何仅从线性最终图像中推断由场旋、画幅平移、填充值、重采样和通道错位造成的不可靠边缘。目标是生成保守的矩形裁切候选和置信度，而不是把像素统计伪装成真实覆盖证据。本项不负责修复坏像素、恢复被裁掉的天空、判断科学目标真伪，也不主张对非线性拉伸图像使用同一组阈值。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 无数据像素：原始数组中的非有限值，或由软件写入的明确空值；这是最强的成片证据，但必须在任何归一化或把非有限值替换为零之前保存。
- 填充值像素：几何变换后写入零、常数或其他哨兵值；若哨兵值未知，仅凭亮度可能与真实暗天空不可区分。
- 低覆盖像素：有数值但参与叠加的帧数少，噪声更大、拒绝统计更差；没有覆盖图时只能由局部噪声和台阶间接推断。
- 拒绝或坏像素：宇宙线、热像素、坏列等被拒绝或插值后的像素；孤立缺陷通常不应触发整边裁切。
- 插值像素：位于旋转或畸变变换的支持域内，数值有限但相关噪声、振铃或色偏可能异常。
- 真实暗背景：低表面亮度天空、暗星云或窄带弱信号，数值低并不等于无数据。
- 真实高梯度或彩色结构：明亮恒星晕、发射星云、渐晕和光害梯度可能接触边界，不能仅凭亮度、梯度或色偏裁切。

#### 缺陷成因 (`defect_causes`)

- 场旋、导星漂移、抖动和子帧构图差异使公共覆盖区缩小，并在叠加外框形成三角形、楔形或阶梯状低覆盖区。
- 配准、畸变校正或放大后的画布填充产生零、非有限值或常数边框；插值核还可能在其内侧留下振铃和相关噪声。
- 不同滤镜或彩色通道采用不同覆盖区，导致单通道缺失、红蓝边和彩色窄带。
- 拒绝、裁切和归一化策略可能把低覆盖边缘变成非零、甚至看似平滑的背景，因此单一黑阈值会漏检。
- 局部坏列、芯片间隙和稀疏的 drizzle 空洞会形成内部不可靠区域；它们不能由四边扫描完整表达。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- **优先级**:
  - 原始非有限值或明确哨兵值及其生成记录
  - 由最终线性图像估计的局部背景中位数、稳健尺度和局部噪声
  - 沿四边的成片近黑比例、分位数、常数游程和边界连通性
  - 从内向外的亮度或噪声台阶、跨尺度一致的梯度
  - 相对于中心参考区的通道残差或色偏
  - 纯视觉判断
- **说明**: 本项只拥有像素统计，不能证明真实参与曝光数。AstroDrizzle 的权重和贡献上下文、Montage 的面积覆盖图都表明辅助平面才是直接证据；像素统计必须被标记为回退推断。

#### 所需输入 (`required_inputs`)

- 未拉伸、未截黑的线性浮点亮度或多通道图像；保留读取时的非有限值掩膜。
- 原始宽高、数值范围、通道顺序、是否已去马赛克和是否已做背景扣除。
- 可选但强烈建议：已知填充值、叠加方式、注册插值核、目标或用户 ROI、当前 WCS。
- 用于比较的中心及多个内部背景网格，而不是只取一个中心框。

#### 算法步骤 (`algorithm_steps`)

- 读取图像时立即保存非有限值掩膜、每通道有限性和原始值；再建立仅供统计的有限浮点副本，不能让补零结果反向充当检测证据。
- 在线性域建立亮度面和各通道残差面。把画面划成多尺度网格，排除明显星点和高纹理格后，用迭代裁剪中位数与中位绝对偏差估计内部背景 B、尺度 S 及空间趋势；中心框只作为一个候选参考。
- 对四边向内生成行列剖面：非有限率、哨兵率、近黑率、低分位数、中位数、稳健尺度、唯一值或常数游程、亮度台阶、噪声台阶和色偏分数。
- 在至少三个尺度上对剖面做中值或高斯平滑，寻找从边向内后趋于稳定的变点。星点只在细尺度产生局部峰，真正边框通常在粗尺度仍连续。
- 把硬无数据和多个软异常二值化，做形态学开闭以去除孤立星点，随后标记连通域；只保留接触图像外边界且面积、边界覆盖长度和跨尺度一致性达标的区域。
- 针对四角检查边界连通楔形。若角部异常存在而整行整列大多正常，优先交给可靠掩膜的最大内接矩形求解，避免两边同步逐像素过裁。
- 为每侧输出首个连续稳定正常位置、各证据分数、总体置信度和候选矩形。硬证据高置信；两个软证据且形状一致为中置信；单一亮度或色偏证据为低置信。
- 裁切后在新边界重新计算同一组指标，同时比较内部背景分布是否保持。指标无改善、异常向内迁移、面积损失超限或 ROI 被碰触时停止并转人工复核。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 由纯像素统计得到的是‘疑似数值可靠掩膜’，不是几何 footprint。几何有效区表示输出像素是否被输入变换覆盖；可靠区还应排除低覆盖、拒绝不足、坏像素和插值支持不足。缺少辅助平面时，二者均无法被像素亮度唯一恢复，应在报告中写明推断来源与置信度。

#### 内部空洞策略 (`internal_hole_policy`)

> 孤立非有限点、坏点、小 drizzle 空洞和不接触外边界的连通域不触发外裁；保留为 mask，并按下游任务决定屏蔽或插值。贯穿全高或全宽的坏列可单独标记，但除非它靠近边缘且裁切损失可接受，不应为了一个内部坏列丢掉整侧视场。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> 首选在注册或叠加时生成直接覆盖证据；像素回退检测则应在叠加完成、仍为线性数据且背景提取、光度校色、反卷积、去星、降噪和拉伸之前运行。Siril 明确指出自动背景方法会拟合包括边缘在内的全图，黑色或不均匀叠加边缘会污染模型，因此裁切必须先于该阶段。若多通道之后还要共同校色或合成，所有通道应先求共同安全矩形再同步裁切。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

> 对贯穿一侧的单调边框可使用四边独立裁切；对场旋三角和不规则角部应从边界连通异常掩膜求最大可靠轴对齐矩形。仅作展示时可允许用户旋转构图，但科学链路不应为节省少量角部再做一次额外插值。马赛克和内部空洞优先保留非矩形 mask，而不是强迫裁成包围盒。

#### 安全保护带规则 (`guard_band_rule`)

> 硬无数据边界至少裁至所有新边界均为有限有效像素；软异常边界取跨尺度变点的内侧置信区间上界，再加由已知插值核支持域决定的腐蚀宽度。元数据缺失时只使用可配置的小护带，并要求增加护带确实改善新边界指标；坐标统一采用半开区间并记录取整方向。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 在优化面积前将目标中心、低表面亮度外晕、参考星、测光孔径及背景采样区表示为必保 ROI。候选矩形必须完整包含硬 ROI；软 ROI 被触及时禁止静默裁切并转人工复核。目标接近边界或星云充满画面时，关闭基于亮度和色偏的自动裁切，只允许硬无数据证据。

#### 验收与回退 (`acceptance_and_fallback`)

- **科学安全**: 只有原始非有限值或明确哨兵值、边界连通、裁后辅助信息同步且 ROI 完整时，可将自动结果视为较高置信的科学安全裁切。
- **处理安全**: 两个以上独立软指标、跨尺度一致、裁后显著改善且损失受限，可用于保护后续背景和卷积处理，但仍保留原图及裁切报告。
- **仅展示**: 依赖视觉构图、旋转矩形、补洞或生成式填充的结果只用于展示。
- **退化**: 指标仍超标、只能检测到单一弱证据或最大损失门禁触发时，保留最小硬裁切并标记退化。
- **人工复核**: 目标触边、星云铺满画面、多通道证据冲突、填充值与天空相似或内部大空洞时转人工复核。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 暗星云、极暗线性窄带图像和真实截黑背景会被近黑率误判；解决方式是使用稳健噪声归一化、空间连通和多证据门禁。
- 渐晕、光害或月光形成平滑亮度趋势，可能被行列台阶误判；应先拟合低阶空间趋势，并要求边缘存在突变而非缓变。
- 亮星、衍射芒、星云细丝或银河尘埃触边会产生梯度和色偏；需源掩膜、多尺度一致性及 ROI 保护。
- 填充画布使用与天空相近的非零常数时，亮度无法区分。PixInsight 工作人员记录过常数画布高于零且与天空背景不可分的实际案例，此时必须降为人工复核。
- 背景已被扣除、拉伸、截黑、降噪或压缩时，直方图和噪声关系被改变；不得复用线性阈值。
- 把非有限值预先替换为零会丢失状态信息，并可能把坏像素、无数据和真实零值混为一类。
- 四边整行统计会稀释小型场旋三角；仅看角部小块又容易被星点污染，因此需结合边界连通二维掩膜。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

> 本检测应优先在去马赛克之后的线性图像运行。若必须裁原始 Bayer 或 X-Trans 数据，裁切原点要保持采样相位，或同步更新相位元数据；Bayer 常见 2×2 周期意味着横纵偏移通常取偶数。多通道成片必须分别检测、取共同可靠矩形，禁止只按亮度通道裁切后让颜色通道具有不同 footprint。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

> 对裁切原点为零基像素偏移 x0、y0 的普通切片，应通过 WCS 库同步平移参考像素并更新 NAXIS；Astropy 的 Cutout2D 会返回为子图更新后的 WCS，其源码对切片执行参考像素减去起点。若存在 SIP 或畸变查找表需额外验证，Cutout2D 文档明确警告其不处理某类畸变查找表。即便本模式没有覆盖图，也应同步裁切现有的 mask、uncertainty、weight、context、rejection 和源目录坐标，不能只裁 SCI。

#### 下游影响 (`downstream_impact`)

> 未清理的低覆盖或填充边会拉低背景估计、制造背景模型坡度或振铃，破坏光度校色的背景和恒星测量，并在反卷积、卷积降噪、去星和拉伸中传播为边缘晕圈。过度裁切则丢失扩展弱信号、参考星和角视场，降低测光及马赛克价值。纯像素回退的核心目标是把高风险边缘排除出全图统计和有邻域依赖的算子，同时把不确定区留在 mask 中。

#### 质量验证 (`qa_and_validation`)

- 保存裁前、候选和裁后诊断图：原图稳健拉伸、硬无数据、各软指标、边界连通域、候选矩形和 ROI 叠加。
- 逐侧报告非有限率、哨兵率、近黑率、亮度标准化台阶、噪声比、色偏和连续正常游程的前后变化。
- 检查内部背景 B、S、星点数和目标 ROI 像素统计没有异常改变；若改变超过校准容差则回退。
- 对保留的参考星做裁前后像素到天球再往返检查，并检查 WCS 四角和中心坐标。
- 用包含暗星云、满幅星云、窄带低信号、强渐晕、亮星触边、非零填充、NaN、场旋楔形和无缺陷控制组的标注集评估逐像素精确率、召回率、过裁面积及漏裁率。

#### 裁切溯源 (`crop_provenance`)

> 记录输入文件摘要、原始和最终尺寸、数据线性状态、原始非有限数量、估计 B 与 S、每项阈值、每侧剖面、连通域、候选矩形、护带、面积与角视场损失、ROI 门禁、置信度、算法和配置版本、人工覆盖及回退原因。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- Siril 1.5 文档把抖动、场旋和不完整重叠列为叠加黑色或不均匀边缘的来源，并明确要求在自动全图背景拟合前干净裁切，这是处理顺序的直接官方证据。
- SExtractor 官方背景算法采用网格化迭代裁剪与稳健众数或中位数，并生成背景噪声图；权重文档说明坏像素会从背景统计排除，坏像素超过网格一半时以邻近有效网格插值。这支持局部稳健参考和无效比例门禁。
- Photutils Background2D 明确区分无覆盖 coverage_mask 与普通坏像素或源 mask，并把非有限值和填充边缘计入排除比例；这说明零亮度本身不是语义。
- Astropy sigma_clip 会自动屏蔽非有限值并可返回掩膜；但项目若先把其替换为零，就失去这项区分。
- PixInsight 工作人员公开案例指出旋转画布的常数填充值高于零且无法与天空背景区分，直接证明纯辐射检测存在不可辨识情形。
- SciPy 连通域标记和二值形态学提供可复现的边界连通、开闭和腐蚀工具；将它们组合为天文裁边门禁属于本调研的工程推断，而不是软件官方自动裁边规范。

#### 直接证据来源 (`evidence_sources`)

- **标题**: Siril 1.5 背景提取文档：自动方法与边缘警告 | **网址**: https://siril.readthedocs.io/fr/latest/processing/background.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.5.0；访问于 2026-08-04
- **标题**: SExtractor 2.24.2：背景建模 | **网址**: https://sextractor.readthedocs.io/en/stable/Background.html | **来源类型**: 官方软件文档 | **日期**: 版本 2.24.2；访问于 2026-08-04
- **标题**: SExtractor 2.24.2：权重与内部方差图 | **网址**: https://sextractor.readthedocs.io/en/latest/Weighting.html | **来源类型**: 官方软件文档 | **日期**: 版本 2.24.2；访问于 2026-08-04
- **标题**: Photutils Background2D：覆盖掩膜与排除比例 | **网址**: https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html | **来源类型**: 官方软件文档 | **日期**: 版本 3.0.0；访问于 2026-08-04
- **标题**: Astropy：迭代标准差裁剪 | **网址**: https://docs.astropy.org/en/stable/api/astropy.stats.sigma_clipping.sigma_clip.html | **来源类型**: 官方软件文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: PixInsight 工作人员关于旋转画布非零填充值的案例 | **网址**: https://pixinsight.com/forum/index.php?threads/2-images-mosaic-question.2553/ | **来源类型**: 官方论坛工作人员答复 | **日期**: 2010-12-27；访问于 2026-08-04
- **标题**: SciPy：二维连通域标记 | **网址**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html | **来源类型**: 官方软件文档 | **日期**: 访问于 2026-08-04
- **标题**: SciPy：二值形态学中的腐蚀和膨胀 | **网址**: https://docs.scipy.org/doc/scipy-1.15.3/tutorial/ndimage.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.15.3；访问于 2026-08-04
- **标题**: Astropy Cutout2D：裁切与 WCS 更新 | **网址**: https://docs.astropy.org/en/stable/api/astropy.nddata.utils.Cutout2D.html | **来源类型**: 官方软件文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: Siril 1.2：注册构图与插值选项 | **网址**: https://siril.readthedocs.io/en/1.2/preprocessing/registration.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.2.6；访问于 2026-08-04

#### 当前项目适配情况 (`current_project_fit`)

- 当前 Stage 2 已使用全分辨率 RGB、中心区域背景、每行列近黑率、中位亮度、色偏、红蓝过量、81 像素滚动中位数、边缘向内稳定游程、角部 5×5 检查和三像素护带，方向与本回退方案部分一致。
- 当前 _to_rgb_float_image 会用 nan_to_num 把 NaN、正负无穷替换为零，再截负值；这会不可逆地混合无数据、真实零值和负背景，必须在转换前单独保存有限性掩膜。
- 当前黑阈值由中心暗背景中位数乘 0.5 后限制在 0.0015 至 0.018，行列和颜色阈值含多项固定常数；它们缺少设备和目标标注集校准，应记录为工程启发式而非科学覆盖判定。
- 当前仅用中心 50% 估背景，暗星云、满幅星云、中心目标和强渐晕会污染参考；建议改为多网格稳健背景与源或纹理排除。
- 当前角部检查同步增加两侧裁量，仍可能为小三角过裁；建议先构造二维边界连通异常掩膜，再求包含 ROI 的可靠矩形。
- 当前 edge_black_ratio 使用四边约 5% 像素并以动态暗阈值统计，是有价值的复检，但它不测覆盖数和噪声台阶；达到 0.03 也不能单独证明边缘科学可靠。
- 当前最多三次补裁、改善不足 0.003 时降级，以及单次每侧最多约 3.5% 的门禁符合保守停止思想；还应加入累计面积、单侧损失、ROI 和 WCS 角面积门禁。
- 当前彩边清理只在状态正常时执行并限制剩余尺寸超过 90%，较保守；但配置中的 stage2_color_artifact_max_crop 未被实际读取，且颜色异常不应在没有亮度或覆盖佐证时单独自动裁切。

#### 实施建议 (`implementation_recommendations`)

- **最高优先级最小改动**:
  - 在 _to_rgb_float_fullres 之前保存 np.isfinite 的逐通道掩膜，避免 nan_to_num 丢失无数据语义；把硬无数据率加入 crop report。
  - 把现有绝对阈值输出为可审计配置，并把自动裁切分成硬证据和软证据；软证据至少二取二或二取三且必须边界连通。
  - 增加累计面积、单侧损失和必保 ROI 门禁；颜色异常不得单独触发高置信裁切。
  - 让 stage2_color_artifact_max_crop 真正控制彩边最大裁切，并把未声明的 stage2_level_artifact_window 加入配置模型。
- **中期增强**:
  - 以多网格迭代裁剪中位数和中位绝对偏差替代单一中心框，输出局部背景及噪声图。
  - 构造二维异常掩膜，加入多尺度亮度与噪声变点、边界连通域、形态学去噪和置信度分类。
  - 建立含典型目标和缺陷的正负样本集，校准阈值并测量过裁面积、漏裁率及对 Stage 3 背景模型的影响。
- **长期架构**:
  - 在 Stage 1 注册和叠加时直接输出 footprint、coverage、weight、rejection 与插值信息；Stage 2 以辅助平面为主、像素统计为回退。
  - 让 SCI、mask、weight、uncertainty、WCS 和裁切 provenance 作为同一个数据产品原子传播，并为科学安全、处理安全、仅展示、退化和人工复核建立明确状态机。


<a id="item-4"></a>
## 4. 裁切几何、最大可靠核心与内部空洞

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 比较深空图像在配准、畸变校正和叠加之后的区域求解策略：四边独立裁切、有效区包围盒、最大轴对齐内接矩形、所有帧共同覆盖区、阈值覆盖区、旋转构图和非矩形掩膜；同时规定 drizzle 空洞、坏列、芯片间隙与断开覆盖的处理。本项关注怎样选出适合后续全图处理或科学测量的可靠核心，不讨论对缺失天空进行内容生成。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 几何未覆盖：没有任何输入 footprint 投到输出像素，覆盖数与权重应为零。
- 部分覆盖：只有部分输入帧贡献，值可能正常但方差更高、拒绝自由度更低。
- 充分覆盖：达到任务定义的最小贡献数、权重或有效曝光时间。
- 被拒绝：原本有几何覆盖，但所有候选值因质量位、宇宙线或异常值拒绝而没有可靠样本。
- 插值支持不足：像素中心位于 footprint 内，但核支持域跨入无数据或低覆盖区。
- 内部空洞：drizzle 采样缺口、坏点簇、芯片间隙、坏列或断开 footprint 留下的无效岛。
- 真实暗像素：有正常覆盖和权重，只是天体信号低；不应因亮度接近零而被几何裁切。

#### 缺陷成因 (`defect_causes`)

- 经纬仪式跟踪或长时序中的场旋把矩形输入 footprint 变为相对旋转的多边形，联合画布四角出现三角或弧形无覆盖区。
- 抖动、导星漂移、子帧构图和子帧筛选改变每个输出像素的贡献帧数，在共同区之外形成低覆盖阶梯。
- 畸变校正和 WCS 重投影使边界成为曲线或不规则多边形，并要求面积或权重修正。
- drizzle 的 pixfrac、输出像素尺度和抖动相位不匹配会造成覆盖不均甚至内部零权重空洞。
- 传感器坏列、芯片间隙、遮挡、DQ 拒绝或所有样本被异常值拒绝会造成内部断裂。
- 多通道或多面板数据 footprint 不同，若独立裁切会破坏后续通道对应和共同 WCS。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- **优先级**:
  - 逐帧几何变换后的 footprint 交集或覆盖计数图
  - DQ 或有效像素 mask 与 rejection mask
  - 逆方差 weight、有效曝光时间和 drizzle context 贡献位图
  - 重投影面积图与输出有限性
  - 最终图像的亮度、噪声和颜色统计
- **说明**: AstroDrizzle 的输出包含科学、权重和贡献上下文扩展，其中 context 用位掩码表示哪些输入图像贡献到每个像素；Montage 为每个重投影图生成像素面积信息并累计为输出覆盖权重。这些都是比成片黑度更直接的区域证据。

#### 所需输入 (`required_inputs`)

- 每帧输入尺寸、有效像素 mask 及到最终输出网格的几何变换，或等价的已叠加 coverage、weight、context、DQ 和 rejection 平面。
- 最终 SCI 图像、输出 WCS、插值或 drizzle 核、pixfrac、输出尺度与参与叠加帧数。
- 科学目标、扩展结构、测光孔径、参考星和背景样区的硬或软 ROI。
- 多通道时每个通道的有效掩膜及共同坐标系。

#### 算法步骤 (`algorithm_steps`)

- 把每帧有效像素 mask 通过与 SCI 相同的几何变换投影到输出网格。几何支持用保守的最近邻或面积覆盖计算，不能用会创造中间有效值的普通亮度插值。
- 累计 coverage C、权重 W、有效曝光时间和贡献上下文；将 DQ、拒绝和非有限性合并。分别构造 V_geom＝有任意几何覆盖，V_common＝所有指定帧均覆盖，V_reliable＝覆盖、权重、拒绝余量和有限性均达门禁。
- 根据注册插值支持域和下游算子的邻域半径，对 V_reliable 做距离门禁或二值腐蚀，得到 processing-safe mask。几何 mask、科学可靠 mask 和处理安全 mask 必须分开保存。
- 对无效 mask 做连通域标记，区分触及外边界的缺口与内部空洞；保留面积、包围盒、细长度、到 ROI 距离及来源位。不要用形态学填洞把真实 no-data 改写成有效。
- 生成候选方案：严格共同区的最大轴对齐矩形、阈值可靠区的最大轴对齐矩形、保留参考帧构图的四边矩形、union 加非矩形 mask，以及可选的单次输出方向优化。
- 对二值可靠 mask 求最大全真轴对齐矩形时，可逐行维护连续真值高度并以单调栈求每行直方图最大矩形，整体复杂度为图像像素数数量级；再加入必须包含硬 ROI、最小宽高、长宽比和边缘裕量约束。该实现是工程算法综合。
- 对已知 footprint 多边形，也可先在天球或统一输出平面求交集，再栅格化并求可靠矩形；畸变较强时不要只用四角包围盒近似曲边。
- 为每个候选计算面积、角视场、最小 coverage 或 weight、内部空洞、ROI 裕量及是否需要额外重采样，按科学模式、处理模式或展示模式选择。
- 应用同一半开矩形切片到 SCI 和所有辅助平面，更新 WCS 与目录坐标；裁后重新确认矩形内无不允许的无效像素、边界有足够处理护带且 WCS 往返一致。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 几何 mask 只回答一个输出位置是否被输入 footprint 触及；它不会说明贡献帧数是否足够、样本是否都被拒绝、权重是否过低或核支持是否完整。可靠 mask 应把 coverage、weight、DQ、rejection、非有限值和科学任务最低信噪要求合并。处理安全 mask 还要在可靠 mask 上按插值、卷积、PSF 或模型邻域向内腐蚀。三者不可合并为单一‘非黑即有效’判断。

#### 内部空洞策略 (`internal_hole_policy`)

- 边界连通无效区域参与外裁几何求解；内部空洞默认保留为 mask，而不是扩大四边裁切。
- 小坏点或窄坏列：科学测量时屏蔽；只在展示副本中用局部插值，并记录插值 mask。
- drizzle 小空洞：优先调整 pixfrac、输出尺度或增加合适抖动输入后重新叠加；不能把后期填洞当作恢复的观测。
- 芯片间隙或大断开区：保留非矩形 mask，或把产品拆成多个连通科学区；若穿过关键 ROI，则判为不满足科学安全。
- 只有在产品规范明确允许忽略小洞时，才可在区域优化阶段容忍洞面积上限；最终科学 mask 仍必须保留原洞。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> 最优做法是在 Stage 1 配准和叠加过程中生成并保留 mask、coverage、weight 和 context，在选定输出 WCS 时一次性确定 framing；Stage 2 在仍为线性图像时，根据这些辅助平面求最终可靠矩形并同步裁切。它必须早于全图背景建模、光度或颜色标定、反卷积、去星、降噪和拉伸。若需改变输出方向，应在注册重采样前决定并只重采样一次，而不是叠加后再旋转裁切。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- **四边独立裁切**: 实现简单且能非对称保留构图，适合每一侧都有近似完整横条或竖条的单调缺陷；对只占角部的场旋三角会为清一个角而损失整行整列，对内部洞无能为力。
- **有效区包围盒**: 包围所有有效像素，适合 union 画布尺寸估计，却通常把无效角和内部洞也包含在矩形内，不能作为可靠核心。
- **最大轴对齐内接矩形**: 直接在可靠 mask 内找全有效矩形，能高效处理旋转角和不规则边界，是矩形后期管线的首选；若只最大化面积可能偏离主体或产生不理想长宽比，因此必须加 ROI、最小尺寸和构图约束。
- **严格共同覆盖区**: 要求所有指定输入都覆盖，最容易解释且适合精密测光或多通道共同处理，但一个偏移异常帧即可大幅缩小视场；应先做帧质量和偏移异常筛选。
- **阈值覆盖区**: 允许覆盖数或权重低于满值但高于任务门限，能在面积和信噪之间折中；必须输出 coverage 并在噪声敏感阶段使用处理安全 mask。
- **联合区加非矩形掩膜**: 最大限度保留马赛克或展示视场，适合支持 mask 的测量和分块算法；对假定每个像素均有效的全图背景拟合或卷积算法不安全。
- **旋转矩形**: 只有把方向选择合并进注册输出 WCS、避免第二次插值时才推荐用于科学产品；对已经叠加的图像再次旋转通常只为构图，归为展示路径。

#### 安全保护带规则 (`guard_band_rule`)

- 从严格无效边界到候选矩形至少留出插值核支持域；若变换有局部尺度变化，取矩形边界附近最大映射支持。
- 后续卷积、反卷积、去星或神经网络算子还需额外邻域，processing-safe mask 的总腐蚀半径取需要连续有效输入的最大半径，而非盲目相加所有阶段。
- 以距离变换直接检查候选矩形每个边界像素到不可靠像素的距离，可处理斜边和内部洞，比固定四边像素更准确。
- 矩形使用左闭右开、上闭下开坐标，向内取整；WCS 和辅助平面使用完全相同的整数切片。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 把目标本体及低表面亮度外延、导星或配准参考星、测光孔径和天区、背景采样区、多通道公共目标区定义为几何约束。硬 ROI 必须完整落在候选矩形且到不可靠区距离不小于任务护带；软 ROI 可进入评分惩罚。若没有任何满足硬 ROI 的科学安全矩形，应保留更大 union 和 mask，或要求重叠加，不能静默删掉目标。

#### 验收与回退 (`acceptance_and_fallback`)

- **科学安全**: 基于 footprint、DQ、coverage 或 weight 的可靠 mask，矩形内没有不允许的洞，满足任务覆盖与 ROI，所有辅助平面和 WCS 同步。
- **处理安全**: 可能不要求全帧共同覆盖，但达到最低权重和处理邻域护带，适合背景、校色和卷积步骤。
- **仅展示**: 允许低覆盖边缘、非矩形 union、旋转和像素插值，但必须与科学母版分离。
- **退化**: 只能从成片推断、辅助平面缺失、矩形内仍有小洞或面积损失门禁触发；下游必须读取 mask 或避开相关阶段。
- **人工复核**: 目标与无效区相交、断开多岛、通道 footprint 冲突、严格共同区被异常帧压缩或需要选择构图方向。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 有效区包围盒把旋转后的无效角和内部洞包含进来，名称上的‘包围有效区’不等于内部全有效。
- 四边扫描面对小三角场旋边会同步多裁两侧；面对曲边、锯齿和断开区域则可能漏掉孤立无效像素。
- 严格所有帧交集会被单个大幅漂移、错误变换或低质量帧支配，先筛帧比事后大裁切更合理。
- 只用覆盖数忽略权重、曝光时间和拒绝会把大量低质量样本与少量高质量样本视为等价。
- 只用权重阈值可能受不同归一化约定影响；应记录 weight 类型并与 coverage 或 context 交叉验证。
- 形态学闭运算若直接作用于科学 mask，会把真实 drizzle 空洞或芯片间隙伪装成有效；只能用于候选形状去噪，原始无效位必须保留。
- 最大面积矩形可能偏离目标、牺牲某一侧构图或变得过长过窄；必须加入 ROI 和构图评分。
- 叠加后再旋转会产生第二次插值、扩展新无数据边界并增加 WCS 风险。
- 多通道各自求最大矩形会得到不同像素网格，造成色边、星点错位和校色样本不一致。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

> 原始 CFA 数据若在去马赛克前裁切，裁切原点必须保持 Bayer 或 X-Trans 相位，或准确更新相位元数据；常见 Bayer 为 2×2 周期，通常要求横纵偏移为偶数。当前项目在 Light 预处理时先去马赛克，因此 Stage 2 主要约束是 RGB 同一矩形。LRGB、窄带和马赛克面板应先把各 mask 映射到统一 WCS，再对需要联合处理的通道取共同可靠区；不同通道不能只共享数值尺寸而不共享天球 footprint。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

- 矩形左上角偏移必须同步到 WCS 参考像素，并更新图像轴长；使用成熟 WCS 切片库可减少坐标原点错误。Astropy Cutout2D 会为切片更新 WCS。
- SIP、多项式或查找表畸变要做专项验证；Astropy Cutout2D 明确警告某些畸变查找表不受支持，不能假定普通参考像素平移覆盖所有情况。
- SCI、mask、DQ、uncertainty、variance、weight、coverage、context、rejection、PSF 图和曝光图使用完全相同的切片；目录像素坐标减去裁切原点，天球坐标保持不变。
- 裁切后用中心、四角、ROI 和随机星点做像素到世界再返回的往返测试；记录坐标约定为零基数组还是一基 FITS。

#### 下游影响 (`downstream_impact`)

> 可靠核心可避免无覆盖和低覆盖边缘污染背景拟合、全局颜色或光度标定、PSF 估计、源检测、反卷积和神经网络补丁；足够的邻域护带还能减少卷积振铃和边缘填充泄漏。内部空洞若未作为 mask 传播，会在背景模型中形成极暗样本、在反卷积中扩散并在拉伸后显著可见。相反，严格共同区或四边过裁会损失弱结构、参考星与拼接余量；因此科学测量可保留 union 加 mask，而只为不支持 mask 的后期阶段另导出 processing-safe 矩形。

#### 质量验证 (`qa_and_validation`)

- 绘制每帧 footprint、coverage、weight、context 位数量、原始和腐蚀后可靠 mask、内部连通域、ROI 以及所有候选矩形。
- 断言 science-safe 矩形内无硬无效像素，processing-safe 矩形边界到无效像素的距离达到护带；统计内部洞数量和最大面积。
- 比较候选的像素与角面积、最小和分位 coverage、最小和分位 weight、每侧损失、ROI 裕量和需要的重采样次数。
- 构造纯平移、已知旋转、强畸变、随机 dither、稀疏 drizzle、坏列、芯片间隙、多连通岛和异常偏移帧的合成测试，并核对理论 footprint。
- 在真实数据上检查裁前后背景模型残差、边缘噪声、星点 PSF、光度差和源检测数量；同时验证 WCS 往返误差。
- 多通道结果要断言尺寸、WCS、裁切原点和 mask 对齐完全一致。

#### 裁切溯源 (`crop_provenance`)

> 记录输入帧清单及变换版本、输出 WCS、原始和最终尺寸、各 mask 定义、coverage 与 weight 类型、阈值、腐蚀半径及其来源、内部空洞连通域统计、所有候选矩形和评分、选中模式、像素和角面积损失、ROI 约束、坐标取整、软件版本、人工覆盖与是否产生额外重采样。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- Siril 官方注册文档提供 current、maximum bounding box、minimum common area 和 center of gravity 四种 framing；minimum 会把序列裁到所有图像共同区域，maximum 则保留完整联合范围。该实践直接对应共同区与 union 的取舍。
- Siril 还说明注册可使用线性、双三次、Lanczos4、面积等插值，并对双三次和 Lanczos4 默认限幅以避免伪影，支持按核区分安全边界。
- Fruchter 与 Hook 的 Drizzle 论文说明该算法可按输入像素统计显著性加权、校正几何畸变，并由 pixfrac 控制 drop 尺寸；较小 drop 与采样相位共同决定覆盖均匀性。
- AstroDrizzle 官方文档说明最终产品可同时包含 SCI、WHT 和 CTX，CTX 以位掩码记录哪些输入图像贡献到每个像素，最终权重可采用曝光、误差或逆方差。
- Montage 官方算法为重投影像素保留面积覆盖并在合并时累计加权，指出真实覆盖变化主要发生在覆盖边缘；这为几何或面积证据优先提供独立实现佐证。
- Photutils 明确区分无覆盖 mask 与普通坏像素 mask；Astropy NDData 或 CCDData 把 data、mask、uncertainty、flags 和 WCS 作为关联平面，支持原子传播思想。
- SciPy 提供连通域标记、腐蚀和膨胀等基础操作。最大全真轴对齐矩形、ROI 约束评分及多状态验收是本调研的工程综合，不是上述天文软件公开的自动裁边内部算法。

#### 直接证据来源 (`evidence_sources`)

- **标题**: Siril 1.2.6：注册构图、共同区和插值方法 | **网址**: https://siril.readthedocs.io/en/1.2/preprocessing/registration.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.2.6；访问于 2026-08-04
- **标题**: Siril 1.5：序列注册与最小共同区域示例 | **网址**: https://siril.readthedocs.io/_/downloads/en/latest/pdf/ | **来源类型**: 官方软件手册 | **日期**: 版本 1.5.0；访问于 2026-08-04
- **标题**: Drizzle：欠采样图像的线性重建方法 | **网址**: https://arxiv.org/abs/astro-ph/9808087 | **来源类型**: 同行评议论文的预印本 | **日期**: 预印本 1998；期刊版本 2002
- **标题**: AstroDrizzle 主接口：科学、权重与贡献上下文产品 | **网址**: https://drizzlepac.readthedocs.io/en/deployment/astrodrizzle.html | **来源类型**: 官方软件文档 | **日期**: 文档构建于 2019-03-08；访问于 2026-08-04
- **标题**: Montage：重投影、面积覆盖、合并与 Drizzle 算法 | **网址**: https://irsa.ipac.caltech.edu/Montage/docs/algorithms.html | **来源类型**: 美国航天局红外科学档案官方软件文档 | **日期**: 访问于 2026-08-04
- **标题**: Montage 组件：重投影图与像素面积图 | **网址**: https://irsa.ipac.caltech.edu/Montage/docs/components.html | **来源类型**: 美国航天局红外科学档案官方软件文档 | **日期**: 访问于 2026-08-04
- **标题**: Photutils Background2D：无覆盖与普通掩膜的区分 | **网址**: https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html | **来源类型**: 官方软件文档 | **日期**: 版本 3.0.0；访问于 2026-08-04
- **标题**: Astropy NDData：数据、掩膜、不确定度与 WCS 容器 | **网址**: https://docs.astropy.org/en/stable/nddata/index.html | **来源类型**: 官方软件文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: Astropy Cutout2D：子图切片和 WCS 更新 | **网址**: https://docs.astropy.org/en/stable/api/astropy.nddata.utils.Cutout2D.html | **来源类型**: 官方软件文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: SciPy：连通域标记 | **网址**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html | **来源类型**: 官方软件文档 | **日期**: 访问于 2026-08-04
- **标题**: SciPy：二值形态学中的腐蚀与膨胀 | **网址**: https://docs.scipy.org/doc/scipy-1.15.3/tutorial/ndimage.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.15.3；访问于 2026-08-04

#### 当前项目适配情况 (`current_project_fit`)

- 当前 Light 路径在 Stage 1 执行去马赛克、两遍注册、seqapplyreg 和拒绝叠加，但 seqapplyreg 未显式选择 minimum common area，也没有将 coverage、weight、context、rejection 或 footprint 作为 Stage 2 输入保存。
- Stage 2 因此完全从最终 SCI 像素推断边缘，无法区分真实暗背景、低覆盖和明确 no-data，也不能识别被填成正常背景的边界。
- 当前四边独立扫描允许非对称裁切，适合条带边框；角部 5×5 while 循环让左上等场旋三角同时增加相邻两侧，可能比最大可靠矩形损失更多。
- 当前只输出矩形裁切总量和各次原因，没有二维可靠 mask、内部空洞连通域、coverage 或处理安全距离，因此无法保证矩形内部全可靠。
- 三像素 guard band 是固定经验值，没有与 Siril 注册插值核或下游最大邻域关联。
- 当前 stage2_corrected.fit 是后续统一输入，矩形产品适合现有 Siril 全图流程；因此短期仍应保留矩形输出，同时新增 mask 和候选几何报告。
- 当前裁切报告记录原始和最终尺寸、每次切片及累计四边裁量，是良好 provenance 基础；需要扩展 WCS、角面积、ROI、候选评分、mask 摘要和辅助平面同步状态。

#### 实施建议 (`implementation_recommendations`)

- **最高优先级最小改动**:
  - Stage 1 显式记录注册 framing、插值方法、参与帧数和每帧变换；若 Siril 能输出或脚本可重建 footprint，生成至少一张 coverage count FITS。
  - Stage 2 同时输出原始有效 mask、候选矩形内是否含洞、每侧及总面积损失和 WCS 更新验证；没有直接 mask 时明确标记 radiometric fallback。
  - 为现有四边裁切加入 ROI、累计面积、单侧最大损失和最小尺寸门禁，避免异常帧或角部检测造成静默过裁。
- **中期增强**:
  - 实现 footprint 或 coverage 到可靠 mask 的构建、边界与内部连通域分类、距离变换护带，以及带 ROI 约束的最大全真轴对齐矩形。
  - 保留两种产物：包含最大视场和 mask 的 science-union，以及供不支持 mask 的后期步骤使用的 processing-safe 矩形。
  - RGB、LRGB 和窄带数据在统一 WCS 上取共同可靠 mask，SCI 与所有辅助平面原子切片。
- **长期架构**:
  - 让 Stage 1 叠加原生输出 SCI、coverage、weight、context、rejection、uncertainty 和 WCS 的产品包，并在输出 WCS 设计阶段一次性优化方向与 footprint。
  - 按科学安全、处理安全、仅展示、退化和人工复核建立状态机；用合成几何和真实标注集持续验证面积损失、漏洞、背景残差、PSF 与 WCS。


<a id="item-5"></a>
## 5. 插值支持域与下游安全保护带

### 研究对象

#### 范围与问题 (`scope_and_question`)

- 回答配准重采样、Drizzle、有限卷积、PSF 匹配、FFT 卷积、反卷积以及图像到图像神经网络会把无效或低覆盖边缘向内影响多远，并给出从有效掩膜推导处理安全区的可复现方法。
- 适用于已经去马赛克的单通道或多通道深空图像、配准后单帧、叠加图及其 coverage/weight/context/mask 辅助平面。
- 不把反射填充、镜像外推、修补或神经网络生成像素视为新增观测；这些方法可以抑制视觉接缝，但不能把边缘升级为科学测量安全像素。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 几何无数据：输出坐标没有任何输入 footprint 覆盖，典型证据是 weight=0、coverage=0、context 无贡献或 NaN。
- 低覆盖：有数值但贡献帧数、有效曝光或逆方差权重低于主体区域，噪声与 PSF 可能不均匀。
- 插值支持不完整：输出像素中心落在几何 footprint 内，但插值核的一部分采样到了无效像素、填充值或图像外部。
- 合成边界上下文：由常数、零、最近值、镜像、周期环绕或修补生成；可用于数值稳定或展示，但来源不是独立观测。
- 算子污染带：卷积、去卷积、FFT 周期假设或神经网络填充使边界误差传播到内部。
- 真实暗背景：有正常覆盖、权重和噪声，仅天区本身暗；不能仅凭接近零的像素值判为无效。

#### 缺陷成因 (`defect_causes`)

- 场旋、平移、畸变校正和尺度改变使各帧 footprint 不重合，输出矩形四角或侧边形成无覆盖/低覆盖区域。
- 有限插值核跨越有效区边界时，把填充值、坏像素或被拒绝像素混入输出；Lanczos 的负旁瓣还会在突变、坏点和宇宙线附近产生扩展波纹。
- Drizzle 的 pixfrac 与输出尺度共同决定输入 pixel drop 覆盖哪些输出像素；pixfrac 太小且 dither 数不足会产生 weight holes，太大则增加卷积与相关噪声。
- FFT 卷积的零填充会产生暗边；不足的 PSF padding 会导致环绕污染；反卷积中的周期边界不连续会产生 ringing。
- 去星、降噪或增强网络在边缘缺少真实上下文；same padding 保持尺寸却不保证边缘预测可靠，含全局注意力或归一化的模型甚至没有简单的有限局部保护带。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- 优先级一：每帧 DQ/坏像素掩膜、配准变换、WCS footprint、叠加 coverage/参与帧数、有效曝光、逆方差 weight、context 与 rejection map；这些信息直接描述数据来源。
- 优先级二：把每个输入有效掩膜用与科学图像一致的几何变换传播，但掩膜应采用布尔覆盖逻辑或保守 footprint，而不是对掩膜做会制造分数有效性的普通亮度插值。
- 优先级三：最终图像中的 NaN、非有限值、接近填充值、噪声突变、亮度台阶和色偏，仅在辅助平面缺失时作降级证据。
- 可靠掩膜必须在几何掩膜之上再叠加覆盖/权重阈值与算子支持域；仅凭矩形内有数值不能证明该像素可供下游处理。

#### 所需输入 (`required_inputs`)

- 科学图像及其当前像素网格、原始有效掩膜或 DQ、coverage/weight/context/rejection 辅助平面。
- 每次配准/重采样的局部变换或 WCS、输入与输出像素尺度、插值核名称和参数、Drizzle 的 pixfrac/scale。
- 后续有限核尺寸或有效非零支持、PSF 图、FFT 边界与 padding 方式、反卷积参数、神经网络结构/理论感受野/切片重叠规则。
- 目标 ROI、参考星与背景采样区，用来判定扩大保护带是否会伤害科学任务。

#### 算法步骤 (`algorithm_steps`)

- 一、构造基础有效掩膜：逐输入合并 DQ、饱和、坏列和拒绝标记，用实际配准变换映射到输出，生成几何覆盖、参与帧数、有效曝光、weight 与 context。
- 二、定义可靠覆盖掩膜：先剔除零覆盖/零权重，再按当前任务的最小参与帧数和相对主体权重阈值筛选；多通道科学产品取各通道可靠掩膜的交集。
- 三、为每个已执行或即将执行的算子生成支持 footprint。重采样使用局部逆变换和插值核；有限卷积使用核非零位置；高斯使用显式 radius；神经网络使用可验证的输入 halo。
- 四、进行掩膜传播：一个输出像素只有在算子所访问的全部输入位置都可靠时才可靠。实现上等价于用该算子 footprint 腐蚀有效掩膜，或用同一 footprint 膨胀无效掩膜。非线性几何变换应逐块使用局部 Jacobian 或直接按采样点求值。
- 五、连续多个局部算子的保护带按有效支持的 Minkowski 和传播；在同一各向同性网格上的保守近似是各半径相加，存在缩放/旋转时先换算到最终输出像素。
- 六、FFT 或反卷积前优先在候选裁切区之外保留工作 halo，并使用足够的 PSF padding；必要时对外边缘 apodization。处理后只交付中心可靠区，填充区及受填充影响区继续保持 mask。
- 七、若矩形裁切会因内部孔洞损失大量视场，则保留非矩形 mask；Drizzle holes 应优先回到 pixfrac、输出尺度、DQ bits 或输入帧组合修复，而不是把孔洞插值后宣称为科学有效。
- 八、生成候选矩形后重新运行计划中的最宽支持算子，在边缘带比较全图处理与扩边/重叠切片处理结果；若差异超过主体噪声容限则扩大 halo 或降级。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

- 几何有效掩膜只回答输出像素中心是否被至少一个输入 footprint 覆盖。
- 可靠掩膜还要求覆盖深度/weight 达标、没有硬 DQ、插值核支持完整，并为后续卷积、PSF、反卷积或模型感受野留出上下文。
- 因此 reliable_mask = geometric_mask ∩ coverage_gate ∩ quality_mask，再按后续算子 footprint 腐蚀；两者不得混为一个‘非黑即有效’条件。

#### 内部空洞策略 (`internal_hole_policy`)

- 对 weight=0 或 coverage=0 的内部孔洞继续保留 mask；局部坏点可在展示副本中修补，但科学主产品不得把修补值标为有效。
- 孔洞密集或出现在 ROI 时，优先回退重叠加：检查 DQ bits、拒绝阈值、pixfrac、输出 scale 和 dither 采样。
- 只有孔洞连到外边且矩形裁切损失可接受时才由裁切消除；坏列、芯片缝和断开岛通常适合 mask，而不是强行裁成很小矩形。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

- 基础 footprint、coverage 和可靠掩膜应在每次配准/重采样与叠加时生成，而不是等到最终图像再猜。
- 用于背景提取、校色、源检测的第一版裁切/掩膜应在叠加后、背景建模前完成；这与 Siril 官方教程把裁边放在背景提取、光度校色和反卷积之前一致。
- 在反卷积、去星、神经网络降噪/增强之前，应从第一版可靠掩膜继续按计划算子腐蚀，或者给处理器额外工作 halo；最终交付裁切在所有可能扩大污染带的算子完成后复检。
- 不要把拉伸后的像素阈值作为主要边界判断，因为拉伸会改变黑度和噪声分布；展示裁切可以在末端另行生成，但必须与科学主产品分离。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- 若下游仅接受矩形，先在被支持域腐蚀后的 reliable_mask 中求满足 ROI 约束的最大/高评分内接轴对齐矩形；场旋明显且允许旋转时可比较旋转矩形，但旋转本身再次需要重采样和新支持传播。
- 若必须保留最大视场或马赛克非矩形 footprint，则保持 union framing 和 mask/weight，不为填满矩形而伪造科学数据。
- 四边独立裁切适合缺陷确实连到外边且近似条带的情形；它不能解决内部孔洞、芯片缝或不规则低覆盖岛。

#### 验收与回退 (`acceptance_and_fallback`)

- science-safe：ROI、参考星与背景环均处于可靠掩膜内，覆盖/权重达标，所有计划算子支持完整，辅助平面和 WCS 同步。
- processing-safe：可安全执行背景、校色、卷积或模型，但覆盖均匀度或修补像素不满足定量科学测量；输出须带 mask。
- display-only：使用镜像/常数 padding、inpainting、生成式补边或允许模型在缺上下文边缘输出；不得回写为科学有效像素。
- degraded：缺少 coverage/weight 或模型支持信息，只能依靠最终图像统计；保留保守 mask、跳过高风险下游算子或扩大人工复核。
- 若 ROI 与无效区冲突，优先重叠加、调整输出 framing/pixfrac/scale 或增加输入帧；不能通过更激进裁切假装问题消失。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 把暗星云、真实低表面亮度天空或平场后的渐晕当作黑边；coverage/weight 可区分‘暗但有数据’与‘无数据’。
- Lanczos 负旁瓣会让边界附近出现负值或亮暗波纹，单纯近黑阈值可能漏检支持域污染，也可能过裁真实负背景。
- 只按像素中心是否覆盖会漏掉插值核跨界；只按非零 weight 又会接受贡献帧极少的高噪像素。
- 把 reflect/mirror padding 当成真实测量，或在 FFT padding 后误以为 same 输出全幅都可靠。
- 用 PSF FWHM 代替完整离散核支持会低估长翼；用理论感受野替代模型实测可能高估或低估有效影响，注意力/归一化还会引入非局部依赖。
- 连续多次重采样却只计算最后一次核半径，或未把输入/输出像素尺度换算到同一网格。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

- 未去马赛克 CFA 的裁切必须保持采样相位：Siril 文档说明 Bayer 裁切会吸附到 2×2 边界，X-Trans 吸附到 6×6 边界；任意角度旋转会破坏 CFA 排列。
- RGB/LRGB/窄带合成的 science-safe 矩形应位于所有必需通道可靠掩膜的交集；允许 union 展示时仍应逐通道保留 mask/weight，不能以另一个通道填补科学权重。
- 各通道 PSF 或重采样核不同，应先分别传播支持，再求共同可靠区。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

- 裁切后同步更新 NAXIS、CRPIX/像素原点及 WCS；旋转/重采样还要更新 CD/PC/SIP 等几何项，并验证天空坐标往返。
- 对 science、mask、uncertainty、weight、coverage、context、rejection 和 PSF 适用图使用完全相同的裁切窗口；布尔/标志平面不得用会混合类别的 Lanczos。
- Siril 自 1.0 起声明 crop、mirror 和 rotate 会更新 WCS；但项目仍应对保存后的 FITS 做像素到天空再返回的自动验收，而不是只信命令成功。
- 记录每次网格变换与支持传播所用参数，使后续阶段能够重建最终可靠掩膜。

#### 下游影响 (`downstream_impact`)

- 背景建模：无效或低覆盖边会拉低统计量，插值相关噪声会改变局部 RMS；应以可靠掩膜排除，且背景网格本身也需要完整支持。
- 校色、星点检测和 PSF：边缘核不完整会改变星峰、FWHM、圆度和通量。Photutils 的 DAOStarFinder 可排除距边界半个卷积核以内的源，说明检测器需要显式边界支持。
- 反卷积：FFT 周期假设、零填充和高对比边界会引发 ringing；edge taper、权重/mask、停止迭代及最终裁掉污染带比简单预裁 3 像素可靠。
- 去星与神经网络：边缘缺少上下文会产生接缝、残星、纹理或颜色幻觉；必须使用 overlap-tile/工作 halo 并丢弃不可信输出。
- 降噪与拉伸：低覆盖区噪声更高，统一强度会过平滑或放大斑驳；即便展示仍保留 weight-aware 门控。
- 测光和源检测：修补/镜像像素、相关噪声及不完整 aperture 会使误差模型失真，因此 science-safe 判定必须比视觉无黑边更严格。

#### 质量验证 (`qa_and_validation`)

- 输出几何有效掩膜、coverage/weight 热图、可靠掩膜、各算子腐蚀后的掩膜及最终矩形叠加图，人工检查 ROI 与低表面亮度结构。
- 在主体区和目标 ROI 统计 weight/coverage 的中位数、离散度、最小值、零权重孔洞数与最大连通孔洞；Drizzle 数据可参考 ROI 内 weight 标准差/中位数小于约 0.2 的经验检查。
- 对每种下游算子做整幅扩边处理与候选裁切/切片处理的重叠区差异图；边缘差异应回落到主体噪声或预先定义的科学误差预算。
- 注入星点或选用真实恒星跨不同边距比较通量、FWHM、椭率、残差与背景 RMS，确定实际安全 halo。
- 裁切前后抽样天空坐标，执行 pixel→world→pixel 往返并核对 CRPIX 平移；所有辅助平面形状、原点和像素索引必须一致。

#### 裁切溯源 (`crop_provenance`)

- 记录原始尺寸/WCS、每个候选矩形、累计四边损失、角视场、输入与输出像素尺度、配准误差。
- 记录基础掩膜来源、coverage/weight 阈值、插值核、pixfrac/scale、每个下游核或模型 halo、组合支持与腐蚀版本。
- 记录 ROI/参考星/背景环版本、候选评分、接受等级、差异验证结果、软件版本、人工覆盖与原因。

### 证据与项目映射

#### 直接证据来源 (`evidence_sources`)

- 《Drizzle：欠采样图像的线性重建方法》，同行评审论文，2002 年，https://doi.org/10.1086/338393
- STScI《Drizzle 概念》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-3-description-of-the-drizzle-algorithm/3-2-drizzle-concept
- STScI《运行 AstroDrizzle：选择最佳 scale 与 pixfrac》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-6-reprocessing-with-the-drizzlepac-package/6-3-running-astrodrizzle
- STScI《检查重处理后的 Drizzle 产品》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-7-data-quality-checks-and-trouble-shooting-problems/7-3-inspecting-drizzled-products-after-user-reprocessing
- AstrOmatic《SWarp 用户手册与软件页》，官方软件文档，访问日期 2026-08-04，https://www.astromatic.net/software/swarp/
- SciPy《高斯滤波》，官方 API 文档，访问日期 2026-08-04，https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
- SciPy《FFT 卷积》，官方 API 文档，访问日期 2026-08-04，https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.fftconvolve.html
- Astropy《FFT 卷积》，官方 API 文档，访问日期 2026-08-04，https://docs.astropy.org/en/latest/api/astropy.convolution.convolve_fft.html
- Photutils《PSF 匹配》，官方文档，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/user_guide/psf_matching.html
- 《U-Net：用于生物医学图像分割的卷积网络》，会议论文与 arXiv 预印本，2015 年，https://arxiv.org/abs/1505.04597
- SciPy《二值腐蚀》，官方 API 文档，访问日期 2026-08-04，https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_erosion.html
- MathWorks《边缘渐消与去模糊振铃》，官方算法文档，访问日期 2026-08-04，https://www.mathworks.com/help/images/ref/edgetaper.html
- Photutils《DAOStarFinder》，官方 API 文档，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
- Siril《主界面：CFA 裁切约束》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- Siril《完整图像处理教程》，官方教程，访问日期 2026-08-04，https://siril.org/tutorials/tuto-scripts/

#### 实施建议 (`implementation_recommendations`)

- 优先级 P0、最小改动：把固定 3 像素改为配置化的算子感知下限。至少记录 Stage 1 重采样核和尺度；Lanczos3 约 1:1 时取 3，加 1 像素取整/配准余量，并与 Stage 5 实际 PSF 半径取更保守组合。保留现有像素统计作为 fallback。
- 优先级 P0：在 stage2_crop_report.json 增加 validity_source、resampling_kernel、input/output_scale、registration_error、planned_operator_support、computed_guard_band、acceptance_class 和 uncertain_reason；若这些元数据缺失，明确标 degraded，而不是静默使用 3。
- 优先级 P1：Stage 1 输出 coverage_count/weight 或至少 common-footprint mask；Stage 2 先按 coverage/weight 建 reliable_mask，再用 SciPy 二值腐蚀传播计划算子支持，最后从腐蚀后掩膜求矩形。edge_black_ratio 只用于一致性检查。
- 优先级 P1：为 Stage 5 及去星/神经网络保留工作 halo，处理后丢弃 halo；新增整图与 overlap-tile 差异测试，用可配置噪声归一化阈值标定模型安全边界。
- 优先级 P1：内部 holes 保留 mask 并触发 Drizzle/叠加建议，不用矩形裁切或像素填补把它升级为科学有效。
- 优先级 P2：建立贯穿 pipeline 的 data+mask+uncertainty+coverage+WCS 数据契约；每次几何和卷积操作都传播这些平面，分别输出 science-safe 主产品与可使用镜像/inpainting 的 display-only 产品。
- 优先级 P2：对每个支持规则增加合成场旋黑角、Lanczos 边界、Drizzle holes、宽 PSF、FFT ringing 和分块神经网络回归样例；以可靠掩膜召回率、ROI 保留率、边缘差异和 WCS 往返误差验收。


<a id="item-6"></a>
## 6. 裁切决策、目标保护与停止条件

### 研究对象

#### 范围与问题 (`scope_and_question`)

- 研究何时应自动裁切、保留掩膜、降级或请求人工复核，重点覆盖 coverage/weight 门、最大视场损失、科学目标 ROI、参考星与天空背景环、迭代收敛和置信度。
- 适用于配准叠加后的线性深空图像，以及为背景提取、校色、反卷积、去星、拉伸、测光或展示生成的不同交付物。
- 不假设存在适用于所有相机、帧数、dither、目标类型和科学任务的固定裁切比例；数值策略必须区分来源事实、项目默认建议和数据集标定值。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 硬无效：非有限、weight=0、coverage=0、无 context 贡献或明确 DQ 禁用。
- 低覆盖：有像素值但贡献帧数、有效曝光或逆方差权重明显低于主体区。
- 被拒绝/坏像素：由 DQ、rejection 或坏点图标明，可能在其他帧有良好替代，也可能最终形成孔洞。
- 插值/填充值：数值存在但依赖图外常数、镜像、修补或生成模型，不等于观测。
- 真实暗背景或暗星云：覆盖和噪声正常但亮度低，是裁切误判的主要风险。
- 任务关键像素：目标及弱信号外延、比较/检验星、测光 aperture、天空 annulus、背景模型采样区和 PSF/校色参考星。

#### 缺陷成因 (`defect_causes`)

- 场旋、dither、平移、尺度和畸变校正使矩形输出边缘的参与帧数下降或完全无覆盖。
- 过小 pixfrac、过细输出 scale、DQ/rejection 过严或输入帧过少会形成 Drizzle holes；坏列、芯片缝可形成内部无效区。
- 最终图像的填充值、插值、Lanczos 波纹、通道错位或卷积边界会制造看似有值但不适合后续处理的边缘。
- 自动算法若只比较中心与边缘亮度，会把真实渐晕、暗星云、扩展星云或非均匀马赛克误判为坏边。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- 最高优先：WCS/配准 footprint、DQ/mask、coverage/参与帧数、有效曝光、inverse-variance weight、context 和 rejection map。
- 其次：多帧贡献一致性、局部噪声与 PSF 稳定度；这些可以识别有覆盖但质量不足的边缘。
- 最后：最终 RGB/FITS 的 NaN、近黑比例、亮度台阶、梯度和色偏；只能在辅助数据缺失时作为降级证据，并须与真实暗结构保护联合。
- 决策时应把证据来源写入置信度，不能让低优先像素统计覆盖高优先 coverage/DQ 事实。

#### 所需输入 (`required_inputs`)

- 叠加科学图、有效掩膜、coverage/weight/context/DQ/rejection、原始和当前 WCS/尺寸。
- 目标 ROI 或 WCS 坐标、扩展弱信号保护区、参考星清单、测光 aperture 与天空 annulus、背景采样需求。
- 候选矩形集合、每边和面积损失、角视场、后续最大算子支持/工作 halo。
- 裁切前后 QA 指标、数据来源置信度、软件与参数版本、人工锁定/覆盖信息。

#### 算法步骤 (`algorithm_steps`)

- 一、从 footprint/DQ/coverage/weight 构造 geometric_mask 与 reliable_mask；缺失时才调用像素统计 fallback，并降低置信度。
- 二、把所有计划下游算子的支持域作用到 reliable_mask，得到 processing-safe mask；硬无效、低覆盖和内部孔洞继续单独标记。
- 三、定义任务约束：主目标和弱信号 ROI、比较/检验星、测光 aperture/天空 annulus、背景网格、PSF 与校色星，并按处理支持扩张。
- 四、从原始未裁网格一次性生成候选矩形：四边扫描、最大内接矩形、共同覆盖区或保留 union+mask；不要在累积裁图上反复丢失坐标。
- 五、为候选计算多目标评分：减少硬无效和低覆盖、提高 coverage/weight 均匀度，同时惩罚面积/角视场损失、ROI 相交、参考星减少和背景网格不足。硬约束优先于分数。
- 六、按置信度与产品用途分类接受：science-safe、processing-safe、display-only、degraded 或 manual-review。
- 七、对已接受候选复算全部指标和 WCS，确保裁切后没有新的核边界风险；若改善不足、ROI 受损或辅助平面不同步则停止并回退。
- 八、保存裁切报告和可视化叠加；人工覆盖必须记录原候选、覆盖值和理由。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

- 几何掩膜表示至少有一个输入 footprint 覆盖，不保证贡献数、噪声、PSF 或插值支持足够。
- 可靠掩膜在几何掩膜上叠加 DQ、coverage/weight、rejection 与算子支持；processing-safe mask 还应为未来卷积、反卷积或模型保留 halo。
- 裁切只能选择一个矩形表达；不规则可靠区和内部孔洞仍需 mask，因此‘已裁切’不等于‘全像素可靠’。

#### 内部空洞策略 (`internal_hole_policy`)

- ROI 内部 weight=0/coverage=0 时禁止标 science-safe，优先回到叠加阶段调整 DQ bits、pixfrac、scale、拒绝策略或补充帧。
- 少量孤立坏点继续以 mask 传播；展示副本可修补但标 display-only。
- 只有与外边连通且矩形裁切损失可接受的孔洞才由裁切消除；芯片缝和马赛克缺口通常保留非矩形 mask。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

- 第一版边界决策在配准叠加完成后、背景提取前执行，避免黑边/低覆盖污染背景统计；Siril 官方教程也是先裁边，再背景提取、光度校色和反卷积。
- 目标 ROI 和产品用途必须在最终自动接受裁切前已知；若目标画像只能在后续确定，则 Stage 2 应保存可逆候选/掩膜并在 Stage 3/4 preflight 后复核，而不是不可逆地假设中心一定不含扩展信号。
- 反卷积、去星或神经网络可能扩大边界污染，处理时保留工作 halo，并在这些步骤后做最终交付复检；拉伸后只允许生成独立 display 裁切。
- 科学测量应尽量在保留 mask/weight 的线性产品上进行，避免用纯展示裁切替代数据质量信息。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- common/science framing：在所有必需帧或通道达到可靠覆盖的交集内求矩形，适合定量背景、校色和测光。
- reference framing：保持参考帧矩形，边缘通过 mask/weight 表示，适合需要固定坐标或最大视场的处理。
- union/mosaic framing：保留所有覆盖区域和非矩形 footprint，必须携带 coverage/weight；不可把空角填黑后当完整矩形。
- 四边独立裁切适合外连通条带；最大内接轴对齐矩形适合场旋黑角；旋转矩形会引入一次额外重采样，只有角视场收益明显且支持重新传播时采用。

#### 验收与回退 (`acceptance_and_fallback`)

- science-safe：高优先证据齐全；ROI/参考星/背景环完整；硬无效为零；coverage/weight 和算子支持达标；WCS 与辅助平面同步。
- processing-safe：边缘足以支持背景、校色或图像处理，但覆盖均匀度、修补像素或误差模型不足以支持定量科学；必须保留 mask/weight。
- display-only：接受 union 空角、镜像/修补/生成填充或视觉裁切；与科学 FITS 分开命名和记录。
- degraded：仅有像素统计、改善不充分或超出自动损失预算；可继续低风险流程但排除高风险边缘，报告残留。
- manual-review：ROI 冲突、暗星云/大尺度弱信号、马赛克不规则 footprint、通道覆盖冲突、参考星不足、候选分数接近或证据互相矛盾。
- 回退优先级：重叠加/修正 DQ-pixfrac-scale → 保留 mask 的更大 framing → 跳过依赖边缘的下游步骤 → 独立 display-only 输出。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 暗星云、弥散星云外延、真实负背景和渐晕被近黑/暗边阈值误裁；必须用 coverage/weight 与多尺度 ROI 反证。
- 中心本身含亮星云时，以中心统计归一化会把正常外部天空判为暗边；马赛克中心也未必代表主体权重。
- 只看平均 coverage 会掩盖局部 holes；只看 weight 非零会接受极低权重；只看单边条带会漏掉旋转三角和内部孔洞。
- 每轮在已裁图重新按百分比裁切会导致总损失超预算且坐标难追溯；必须相对原图累计。
- 最大面积矩形可能裁掉科学关键但面积很小的目标或参考星，因此 ROI 是硬约束，面积仅是次级目标。
- 把人工修补或镜像 padding 当作覆盖改善，或把像素统计变好误报为信息增加。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

- 未去马赛克 Bayer/X-Trans 必须按采样周期保持相位；Siril 当前文档分别约束到 2×2 和 6×6 边界。
- RGB、LRGB、窄带科学输出的 ROI、参考星和 sky annulus 必须在全部必需通道的可靠掩膜交集中；通道 union 只可用于带逐通道 mask 的展示或特定分析。
- 通道错位和不同 PSF 会使各通道保护带不同，应分别传播后再求共同矩形。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

- 矩形裁切后 NAXIS 与像素原点改变，WCS 至少要更新 CRPIX；旋转/重采样还需更新线性变换和畸变项。Astropy Cutout2D 可返回裁切后更新的 WCS。
- science、mask、uncertainty、weight、coverage、context、rejection、PSF 有效区和 ROI 坐标使用同一窗口；Astropy NDData 的切片示例会同步 data、mask、uncertainty 与 WCS，可作为数据契约参考。
- 保存前后做 pixel→world→pixel 往返，并确认目标、参考星和候选角点的天区位置不变；人工 ROI 应同时保存原图像素坐标与天空坐标。
- Siril 现代版本声明 crop 会更新 WCS，但 pipeline 仍需验证实际保存产物和辅助平面，而不是把软件行为当作完整传播契约。

#### 下游影响 (`downstream_impact`)

- 背景建模：低覆盖/黑边会拉低背景与 RMS，过裁则减少天空样区并可能让扩展星云主导剩余画面。
- 校色与 PSF：参考星数量和空间分布受裁切影响，边缘卷积不完整会偏移通量、FWHM、圆度和颜色。
- 反卷积/去星/降噪：过紧裁切不给算法上下文，会重新制造 ringing、接缝或纹理；应保留工作 halo。
- 拉伸：低覆盖和插值噪声会被显著放大；display-only 可以更积极裁切，但不能替换科学主产品。
- 测光：不完整 aperture 或 sky annulus、修补像素和相关噪声会使通量及误差失真；必须使用 science-safe mask 与有效 overlap 面积。
- 源检测：边缘卷积核不完整会产生漏检/假检；应按检测核排除边界源或标志。

#### 质量验证 (`qa_and_validation`)

- 生成原图、coverage/weight、hard-invalid、reliable-mask、ROI/参考星/天空环和候选矩形的同坐标叠加图。
- 裁切前后对比：零覆盖面积、低覆盖面积、weight 标准差/中位数、最大孔洞、edge_black_ratio、背景 RMS、星点数量/分布、ROI 保留率、角视场和每边损失。
- 对 Drizzle 产品在目标 ROI 单独检查 weight holes 与离散度；对 Background2D 类下游检查每个背景 box 的未遮罩比例，而非只看整图平均。
- 对受边界影响的星做 aperture/PSF 通量、FWHM 和残差对比；对扩展目标在弱拉伸和多通道图中人工核对外延。
- WCS 抽样往返误差、各辅助平面 shape/origin 一致性、候选相对原图的坐标重建必须通过。
- 回归集至少包含场旋黑角、暗星云、大星云铺满画面、低帧数 Drizzle holes、宽边渐晕、通道错位、马赛克和目标贴边。

#### 裁切溯源 (`crop_provenance`)

- 原始/最终尺寸和 WCS、候选矩形、相对原图累计四边损失、角视场与面积保留率。
- 证据源、coverage/weight/DQ 阈值、支持域/保护带、ROI 和参考星版本、候选评分、置信度与接受等级。
- 每轮指标、停止原因、回退路径、软件/配置版本、人工覆盖人/时间/理由及 display/science 产品关联。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- 事实：AstroDrizzle 输出 science、weight 和 context；weight 近似有效曝光，context 记录哪些输入图像贡献到像素，官方 QA 要检查目标附近权重与 holes。
- 事实：DrizzlePac 指出 pixfrac/scale 没有适合所有观测和科学目标的唯一设置；较小 pixfrac 可提高分辨率并降低相关噪声，但降低低表面亮度结构的可见性。
- 事实：Photutils Background2D 明确区分无覆盖 coverage_mask 与源/坏点 mask，并按背景 box 的遮罩比例排除不可靠网格。
- 事实：Photutils aperture 工具支持 mask 和部分 overlap 的有效面积；AAVSO 指南要求目标、比较/检验星和足够天空 annulus。
- 事实：Astropy Cutout2D 在输入 WCS 时返回更新后的裁切 WCS；NDData 将 data、mask、uncertainty、WCS 作为同一数据对象的关联属性。
- 事实：Siril 官方教程把裁边列为背景提取之前的第一项后处理，并说明黑带会扭曲后续统计；同时提示宽场星云可能使背景提取难以判断。
- 推断：将上述来源组合成带 ROI 硬约束的多目标裁切评分、分级接受和停止门，是工程整合方案；官方资料没有给出跨软件统一的最大视场损失阈值。

#### 直接证据来源 (`evidence_sources`)

- STScI《AstroDrizzle 在管线中的产品结构》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-5-drizzlepac-software-package/5-3-astrodrizzle-in-the-pipeline
- STScI《检查重处理后的 Drizzle 产品》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-7-data-quality-checks-and-trouble-shooting-problems/7-3-inspecting-drizzled-products-after-user-reprocessing
- STScI《运行 AstroDrizzle》，官方文档，访问日期 2026-08-04，https://hst-docs.stsci.edu/drizzpac/chapter-6-reprocessing-with-the-drizzlepac-package/6-3-running-astrodrizzle
- Photutils《二维背景估计》，官方 API 文档，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html
- Photutils《孔径测光》，官方用户指南，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/user_guide/aperture.html
- Photutils《像素孔径有效重叠面积》，官方 API 文档，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/api/photutils.aperture.PixelAperture.html
- Photutils《DAOStarFinder》，官方 API 文档，访问日期 2026-08-04，https://photutils.readthedocs.io/en/stable/api/photutils.detection.DAOStarFinder.html
- AAVSO《单色 CCD/CMOS 测光指南》，专业协会指南，2022 年版，https://www.aavso.org/sites/default/files/publications_files/ccd_photometry_guide/CCDPhotometryGuide.pdf
- Astropy《Cutout2D》，官方 API 文档，访问日期 2026-08-04，https://docs.astropy.org/en/stable/api/astropy.nddata.utils.Cutout2D.html
- Astropy《N 维数据及关联 mask、uncertainty、WCS》，官方文档，访问日期 2026-08-04，https://docs.astropy.org/en/stable/nddata/index.html
- Siril《完整图像处理教程》，官方教程，访问日期 2026-08-04，https://siril.org/tutorials/tuto-scripts/
- Siril《主界面：CFA 裁切约束》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- Siril《1.0 发布说明：几何操作更新 WCS》，官方发布说明，2021-11-20，https://siril.org/download/2021-11-20-siril-1.0.0-rc1/

#### 实施建议 (`implementation_recommendations`)

- P0 最小改动：增加总损失计算和硬停止，至少报告每边、任一维与面积损失；先把 5%提示、10%维度或20%面积人工复核作为标注为实验性的可配置值，不直接静默接受。
- P0：在 Stage 2 前读取可用的 FITS/WCS 目标位置和保守目标半径；若 target profile 尚未冻结、检测到大尺度弥散结构或候选接近保护区，输出 manual-review/degraded 并停止进一步自适应裁切。
- P0：把 stage2_color_artifact_max_crop 真正接入实现，并让所有初次/自适应/彩边裁切共享相对原图的总预算；停止条件同时检查 ROI、星点和背景可用面积。
- P1：Stage 1 输出 coverage_count/weight/common-footprint mask；Stage 2 用它作为主证据，像素统计改为 fallback。报告 evidence_source、confidence、acceptance_class 和 remaining_risk。
- P1：在 Stage 3/4 preflight 冻结目标后复核 Stage 2 候选。若需改变，应从 stage1_prepared 基线重算一次裁切，避免在已裁图继续叠加；或者 Stage 2 先只产出 mask/候选，确认后再物理裁切。
- P1：构造 science_roi.json，包含主目标/弱信号多边形、参考/检验星、测光 aperture/annulus 与背景安全区；矩形求解把这些设为硬约束。
- P1：把停止判据升级为多指标：hard-invalid 减少、coverage/weight 均匀度改善、ROI 完整、参考星与背景网格充足、边缘卷积差异降低；不再只依赖 edge_black_ratio 0.003。
- P2：支持 common/reference/union 三种 framing 和 science-safe/processing-safe/display-only 三类产物；内部 holes 由 mask 表示，不能靠展示修补升级。
- P2：建立包含暗星云、满幅星云、目标贴边、低帧数 Drizzle、马赛克和通道错位的标注回归集，分别标定 coverage、损失与收敛阈值。


<a id="item-7"></a>
## 7. 多通道、CFA、马赛克与产品目的

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 研究深空后期中 RGB、LRGB、窄带和多面板马赛克在边缘裁切时怎样定义共同支持区，怎样处理 Bayer 与 X-Trans 原始 CFA 相位，以及 reference、common、union 等画幅策略如何随科学测量、全图处理和展示用途改变。本项不讨论各滤镜的物理通带标定，也不把显示用修补视为观测数据恢复。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 单通道有数据：某滤镜或颜色通道有足够覆盖，但其他待联合通道可能无覆盖或低覆盖。
- 通道共同有效：所有参与像素级组合的通道在同一输出 WCS 像素上均通过覆盖、权重、DQ 和有限性门禁。
- 通道共同低可靠：各通道都有数值，但至少一个通道覆盖、权重或拒绝后样本数不足；颜色比值和窄带映射会被最弱通道支配。
- CFA 几何采样：未去马赛克像素只测一个颜色；Bayer 与 X-Trans 的周期相位属于数据语义，不能把相邻位置当作完整 RGB。
- CFA drizzle 稀疏覆盖：输出位置可能有总贡献却缺少某个颜色的贡献，必须按颜色分别判断覆盖。
- 马赛克 union：至少一个面板覆盖；边缘和面板接缝处常有不同有效曝光时间。
- 马赛克 common：所有指定面板或全部指定通道同时覆盖；对大范围马赛克通常很小、为空或没有业务意义。
- 真实暗信号：在所需通道都有正常权重的暗星云、发射线弱区或暗背景，不应因亮度低而裁掉。

#### 缺陷成因 (`defect_causes`)

- 不同滤镜在不同夜晚、旋转角、dither、焦距或配准参考下采集，导致各通道 footprint、边缘覆盖和星点位置不完全一致。
- 分别叠加 L、R、G、B 或窄带后再各自自动裁切，会得到不同原点和尺寸，逐像素合成时产生色边、空值或隐式再次重采样。
- CFA 图像若从奇数像素原点裁切、错误翻转或旋转，会改变颜色阵列相位；以旧 Bayer pattern 去马赛克会把颜色解释错位。
- Bayer drizzle 或 X-Trans drizzle 的各颜色采样密度不同；抖动不足、放大尺度过高或 pixfrac 较小时可形成颜色相关空洞和边缘噪声。
- 多面板马赛克的目标就是保留不规则 union；以四边全有效为唯一标准会为清除角部或接缝低覆盖而删除完整面板。
- reference framing 受参考帧构图支配，maximum 或 union framing 受所有输入外包络支配，minimum 或 common framing 受交集及异常偏移帧支配。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- **优先级**:
  - 每个通道或滤镜在统一输出 WCS 上的几何 footprint、coverage、weight、DQ 和 rejection
  - CFA drizzle 时按输出颜色分开的 coverage 或 weight，而不是三通道求和
  - 输入 CFA pattern、ROWORDER、XBAYROFF、YBAYROFF、传感器类型及已执行几何变换记录
  - 跨通道恒星配准残差、共同源匹配和 PSF 差异
  - 最终亮度、近黑比例和边缘色偏统计
- **说明**: 颜色组合是否安全首先是同一天球位置上所需通道是否都有可靠观测，而不是合成图是否看起来不黑。最终像素统计只能作为缺少辅助平面时的降级证据。

#### 所需输入 (`required_inputs`)

- 每个 RGB、L、窄带或马赛克面板的 SCI、WCS、几何 footprint、coverage 或 weight、DQ 或 mask、rejection 及曝光信息。
- 通道或面板之间的配准变换、参考图选择、输出 WCS、输出尺寸、像素尺度和方向。
- 未去马赛克时的 CFA 类型与 pattern、周期、ROWORDER、XBAYROFF、YBAYROFF，以及裁切、翻转、旋转和 drizzle 历史。
- 产品目的、需要联合处理的通道集合、科学目标和弱信号 ROI、允许保留的非矩形 mask。

#### 算法步骤 (`algorithm_steps`)

- 先定义产品合同：单通道科学产品、多通道共同处理产品、马赛克科学 union，或仅展示产品；明确哪些通道必须在每个像素共同存在。
- 把所有通道或面板的 SCI 与有效辅助平面重投影到一个已选定的输出 WCS。输出 WCS 应在注册或叠加阶段一次确定，避免为裁切再次旋转插值。
- 为每个通道构造 V_channel：合并 footprint、有限性、DQ、coverage、weight、rejection 和该通道噪声门禁；CFA drizzle 则为 R、G、B 分别构造。
- 构造 V_union＝任一所需通道或面板可靠，V_common＝所有参与像素级组合的通道可靠；另对每个科学任务构造 V_required，例如 LRGB 合成要求 L/R/G/B，HOO 映射要求 Hα/OIII。
- 按配准插值核、PSF 匹配核和下游邻域腐蚀 V_required，得到 processing-safe mask；在其内求受 ROI 和构图约束的最大矩形，或选择参考图画幅与 mask。
- 单视场多通道默认导出同一 WCS、同一整数矩形的共同处理产品；同时可保留每通道更大的 science-union 母版和各自 mask。
- 多面板马赛克默认保留 union、coverage 与非矩形 mask；只为不支持 mask 的全图算法另求 processing-safe 子矩形，不要求所有面板共同覆盖。
- 未去马赛克 CFA 若必须裁切，原点与边界按阵列周期对齐并同步 pattern offset；任意角旋转或缩放推迟到去马赛克或 CFA drizzle 输出之后。
- 裁后对所有联合通道使用完全相同的切片，复检尺寸、WCS、共同 mask、颜色边缘和 ROI；展示副本中的填补、羽化或构图旋转另行记录。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 多通道几何共同区只表示各 footprint 都触及某位置，不保证某通道有足够权重、拒绝后样本数或完整插值支持；数值共同可靠区还要通过逐通道 coverage、weight、DQ、rejection、有限性和噪声门禁。对 CFA drizzle，总几何覆盖也不等于 R、G、B 都有覆盖。马赛克 union 几何 mask 则有意允许非矩形边界和覆盖阶梯，不能因不是全矩形而统称为坏边。

#### 内部空洞策略 (`internal_hole_policy`)

- 任一必需通道的内部洞都从该产品的共同可靠 mask 中排除；洞不应靠裁完整行列隐藏。
- 单通道独有的小坏点可留在该通道 DQ 中；合成时传播为共同 mask。展示副本可做局部插值但保留插值来源位。
- CFA drizzle 的单色空洞必须按颜色记录，不能因其他颜色有贡献而当作有效彩色像素。
- 马赛克芯片间隙或面板之间无覆盖区保留为非矩形 mask；若算法不支持洞，改用分块或安全子矩形。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> 先在校准后的帧或各通道主图上完成天球对齐，选定所有联合产品共享的输出 WCS，并在配准/叠加时生成逐通道 coverage 与 mask。最终共同矩形应在线性阶段、背景建模和任何跨通道比值、颜色合成、PSF 匹配、反卷积、去星与拉伸之前确定。原始 CFA 裁切若不可避免，应在去马赛克前按 CFA 周期执行；构图旋转应合并进注册输出 WCS，或仅在展示末端执行。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- **参考图画幅**: 以选定参考图的 WCS、尺寸和方向为输出网格，构图稳定且便于多通道一致；参考图外仍须按通道保留无覆盖 mask，不能把填充值当作观测。
- **严格共同区**: 对所有需要联合的通道取可靠 mask 交集，适合像素级颜色比、统一背景或不支持 mask 的卷积；最弱通道和异常偏移会决定损失。
- **阈值共同区**: 允许每通道达到各自最低 coverage 或 weight，而不要求所有输入帧满覆盖，通常比逐帧严格交集实用；门禁必须按通道而非合成亮度。
- **联合区**: 保留任一通道或面板的最大天区和辅助 mask，适合单通道科学测量、目录生成与马赛克母版；不适合无 mask 的像素级颜色运算。
- **马赛克画幅**: 通常由期望天球区域或全部面板 union 定义，允许非矩形 footprint、接缝和覆盖阶梯；共同区应指每个天空位置所需滤镜共同有效，不是要求所有面板重叠。
- **每通道独立画幅**: 可作为单通道科学母版，但在 RGB/LRGB/窄带合成前必须重投影和统一裁切；不能仅把数组缩放到同尺寸。

#### 安全保护带规则 (`guard_band_rule`)

- 共同 mask 先按每个通道自身的注册核支持腐蚀，再按跨通道 PSF 匹配或联合模型的最大邻域腐蚀。
- 未去马赛克裁切使用 CFA 周期网格：Bayer 选择框对齐 2×2，X-Trans 对齐 6×6；若修改原点而不保持相位，必须正确更新 pattern offset。
- 候选矩形边界向内取整，所有通道使用同一半开切片；因 CFA 周期取整增加的损失须计入报告。
- 马赛克 mask 的斜边和洞用距离变换定义保护带，不用固定四边像素代替。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 先在天球坐标定义目标核心、低表面亮度外延、测光目标与比较星、背景环或采样区，再映射到每个通道 mask。共同处理矩形必须包含硬 ROI 且 ROI 在所有必需通道有足够支持；若某窄带缺少目标外延，不能以其他通道亮度替代。马赛克的面板外延本身可能就是科学 ROI，此时应保留 union+mask 而不是强求一个小共同矩形。

#### 验收与回退 (`acceptance_and_fallback`)

- **科学安全**: 保留每通道 SCI、WCS、mask、coverage、weight 和不确定度；单通道测量可使用各自可靠区，多通道测量明确所需通道交集，CFA 相位和来源完整。
- **处理安全**: 所有待联合通道在同一 WCS 与矩形内通过最低可靠度，并具有注册、PSF 匹配和后续邻域护带。
- **仅展示**: 允许 union 边缘、羽化、填洞、构图旋转和局部修补，但与线性科学母版分离且不得回流测光。
- **退化**: 只有合成像素统计、缺失逐通道 coverage、CFA 元数据不完整或共同矩形内仍有通道空洞；限制需要颜色比或精密测量的下游步骤。
- **人工复核**: 目标 ROI 与某必需通道无覆盖相交、马赛克多连通、通道间 WCS 残差过大、CFA 相位不明或共同区损失过高。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 把窄带中真实低信号、暗星云或滤镜间物理颜色差异当作黑边或色边。
- 从最终 RGB 色偏反推某通道无效，可能混淆真实发射线结构、渐晕、光害梯度和配准误差。
- 各通道独立自动裁切后只对齐尺寸，不对齐 WCS 和原点，会形成隐蔽的像素对应错误。
- 严格取所有帧或所有面板交集会被异常偏移帧拖垮，也会把真正的马赛克 union 裁成很小区域。
- reference framing 可能在参考图边缘保留其他通道无覆盖像素；maximum framing 会扩大低覆盖边；二者都不能替代 mask。
- 只看总 drizzle coverage 会漏掉红色或蓝色零覆盖，尤其在 CFA 稀疏采样和上采样时。
- 预去马赛克裁切原点奇偶错误、上下翻转后 ROWORDER 未同步，或对 X-Trans 误用 2×2 规则，会产生全图颜色伪影。
- 展示用修补若覆盖原 mask 并作为科学产品保存，会让缺失观测看似正常。
- 多通道共同矩形可能删除某单通道独有但有科学价值的外延，因此应同时保留各自 science-union 母版。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

- **Bayer**: Siril 1.5 官方界面把 CFA 裁切框吸附到 2×2 边界，并可在 x、y 各移动最多 1 像素以保持 pattern 起点；FITS 的 XBAYROFF、YBAYROFF 与 ROWORDER 会改变 pattern 解释。
- **X-Trans**: Siril 将裁切框吸附到 6×6 边界，并可在各方向移动最多 5 像素；不能沿用 Bayer 的偶数偏移规则。
- **几何变换**: 未去马赛克 CFA 不允许任意角旋转或普通缩放；应先去马赛克或完成 CFA drizzle。翻转也必须同步行序与 pattern 相位。
- **单次彩色相机**: 去马赛克后的 RGB 通道共享一个几何网格，但边界插值可靠性仍可能按颜色不同；统一裁切且以最弱颜色支持为门禁。
- **LRGB与窄带**: 各主图必须共享 WCS、像素尺度、方向和整数切片；像素级组合使用必需通道可靠 mask 交集，单通道分析可保留更大区域。
- **CFA drizzle**: R/G/B 贡献应分别统计；红蓝采样更稀疏，Siril 官方说明放大尺度会需要更多输入才能保持覆盖和噪声。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

- 所有通道和面板先映射到共同 WCS；共同裁切对 SCI、逐通道 mask、uncertainty、weight、coverage、context 和 rejection 使用完全相同的整数切片。
- 更新 NAXIS 与 WCS 参考像素，保留天球坐标不变；目录像素坐标按裁切原点平移。多通道必须断言 WCS 与数组 shape 一致。
- 未去马赛克 CFA 同步保存 pattern、ROWORDER、XBAYROFF、YBAYROFF 和裁切原点；若裁切保持完整周期，相位应不变，仍需在文件头与日志中核验。
- 马赛克产品保留面板或曝光贡献 context、逐像素 coverage/area 与权重，不能只保存填成零的 RGB 图。

#### 下游影响 (`downstream_impact`)

> 共同且带保护区的多通道矩形可避免 RGB 比值、LRGB 合成、窄带映射、光度校色、PSF 匹配、去星和卷积被缺失通道或低覆盖边缘污染。逐通道 union+mask 则能保留单滤镜测量和马赛克外延。错误 CFA 相位会在去马赛克后扩散成周期性色伪影，无法靠后续裁边补救；通道独立裁切或额外重采样会降低星点重合与 PSF。背景建模应读取 coverage mask，展示拉伸与修补必须在科学路径之外。

#### 质量验证 (`qa_and_validation`)

- 叠加显示每通道 footprint、coverage、weight、共同 mask、union mask、候选矩形和科学 ROI，并检查边缘与接缝。
- 断言联合产品所有通道的 shape、WCS、像素尺度、方向和裁切原点一致；在中心、四角和随机星点做天球坐标往返与跨通道残差测试。
- 分别统计 R/G/B/L/窄带在矩形内的零权重率、低分位权重、内部空洞和边缘噪声；CFA drizzle 不得只验总 coverage。
- 用四种 Bayer pattern、ROWORDER 上下方向、全部 XBAYROFF/YBAYROFF 组合及 X-Trans 相位做彩色棋盘测试，裁后去马赛克结果应与原图对应区域一致。
- 比较裁前后共同星的质心、FWHM、颜色比和背景残差；马赛克检查面板接缝、覆盖阶梯和角面积。
- science-safe 与 display-only 分别导出，检查显示用填补没有清除科学母版 mask 或覆盖 provenance。

#### 裁切溯源 (`crop_provenance`)

> 记录产品目的、必需与可选通道、输入文件和滤镜、参考图、输出 WCS、各通道 footprint 与 coverage 定义、common/union 候选、最终像素及天球矩形、每通道面积和权重损失、ROI、CFA 类型/pattern/行序/偏移、drizzle 参数、重采样次数、保护半径、软件版本、人工选择及科学或展示等级。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- Siril 1.5 官方注册提供参考图、最大外包框与最小共同区域等 framing：最大模式保留 union，最小模式裁到序列共同区，说明画幅选择本身就是覆盖与构图取舍。
- Siril 1.5 官方主界面明确：Bayer CFA 裁切吸附到 2×2，X-Trans 吸附到 6×6；任意角旋转会破坏 CFA，应先去马赛克或 Bayer drizzle。
- Siril FITS 文档说明 ROWORDER、XBAYROFF 和 YBAYROFF 共同影响 Bayer pattern 的读取，支持将相位视为必须传播的元数据。
- Siril drizzle 文档说明 CFA drizzle 按当前像素颜色落入输出通道；红蓝采样较稀，放大尺度需要更多输入保证覆盖与噪声。
- STScI 单次访问马赛克流程为同一探测器的各滤镜产品定义共同 metawcs，覆盖全部输入视场，并为每个产品保存 SCI、WHT 和 CON；这是跨滤镜统一网格且保留贡献信息的强实例。
- STScI Bubble Nebula 实例先用一个滤镜构建参考马赛克，再将其他滤镜曝光对齐到该马赛克，体现 reference mosaic 的实际用法。
- Montage 在统一天球投影上按像素天空重叠面积重投影和加权，并输出 area coverage；它支持保留马赛克 union，而不是用矩形黑边检测代表覆盖。
- science-safe、processing-safe、display-only 分级及按任务构造 V_required 是本调研的工程综合，并非这些软件采用相同名称公开的标准。

#### 直接证据来源 (`evidence_sources`)

- **标题**: Siril 1.5 主界面：CFA 裁切边界与旋转警告 | **网址**: https://siril.readthedocs.io/en/latest/GUI/main-interface.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.5.0；访问于 2026-08-04
- **标题**: Siril 1.5 FITS：ROWORDER、XBAYROFF 与 YBAYROFF | **网址**: https://siril.readthedocs.io/en/latest/file-formats/FITS.html | **来源类型**: 官方软件文档 | **日期**: 版本 1.5.0；访问于 2026-08-04
- **标题**: Siril 1.5 注册：reference、maximum 与 minimum framing | **网址**: https://siril.readthedocs.io/_/downloads/en/latest/pdf/ | **来源类型**: 官方软件手册 | **日期**: 版本 1.5.0；访问于 2026-08-04
- **标题**: Siril 1.4 Drizzle 与 CFA drizzle | **网址**: https://siril.readthedocs.io/en/stable/preprocessing/drizzle.html | **来源类型**: 官方软件文档 | **日期**: 稳定版 1.4；访问于 2026-08-04
- **标题**: DrizzlePac 单次访问马赛克：跨滤镜共同输出 WCS 与辅助平面 | **网址**: https://ssb.stsci.edu/hack/astrometry/testing/drizzlepac_v341_html/singlevisit.html | **来源类型**: 空间望远镜科学研究所官方软件文档 | **日期**: DrizzlePac 3.2 开发文档；访问于 2026-08-04
- **标题**: HST Bubble Nebula：使用参考滤镜马赛克对齐其他滤镜 | **网址**: https://archive.stsci.edu/prepds/heritage/bubble/README.html | **来源类型**: 空间望远镜科学研究所数据处理说明 | **日期**: 2016；访问于 2026-08-04
- **标题**: Montage 算法：天球重投影、面积覆盖与加权合并 | **网址**: https://irsa.ipac.caltech.edu/Montage/docs/algorithms.html | **来源类型**: 美国航天局红外科学档案官方软件文档 | **日期**: 访问于 2026-08-04
- **标题**: Photutils Background2D：马赛克无覆盖 mask 与普通坏像素 mask | **网址**: https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html | **来源类型**: 官方软件文档 | **日期**: 版本 3.0.0；访问于 2026-08-04

#### 当前项目适配情况 (`current_project_fit`)

- 当前项目的 Light 路径在 Stage 1 使用 Siril calibrate -debayer 后再 register、seqapplyreg 和 stack，因此 Stage 2 面对的是已去马赛克 RGB，通常不再受 CFA 2×2 或 6×6 相位限制。
- 当前 Stage 1 没有显式选择 minimum/common framing，也没有保存逐通道或逐帧 coverage、weight、context 和 rejection 产品；Stage 2 只能从合成 RGB 像素推断边缘。
- Stage 2 对当前 RGB 图只执行一个矩形 crop，三个通道数组保持同尺寸；这是单个 OSC RGB 流程的合理最小行为，但不能证明三色边缘同样可靠。
- 当前色边清理以四周 2.5% 条带相对中心的 RGB 色偏判断，可能把真实窄带结构、渐晕或通道噪声差异误当伪影；它也不等价于逐通道共同 coverage。
- 项目尚无 LRGB、独立窄带主图或多面板马赛克的产品模型；若未来直接复用 Stage 2，各通道独立裁切和马赛克四边裁切会破坏共同 WCS或过度损失 union。
- Stage 2 输出 stage2_corrected.fit 与像素裁切报告，但未记录产品目的、通道集合、共同/联合 mask、角面积或 CFA provenance。
- Stage 4 才进行 platesolve；因此 Stage 2 一般没有新解算的共同 WCS 可用。若输入叠加图已有 WCS，Siril 官方自 1.0 起称 crop 会更新 WCS，但项目仍未做裁后跨通道或往返验证。

#### 实施建议 (`implementation_recommendations`)

- **最高优先级最小改动**:
  - 在 Stage 2 报告新增 product_mode，当前固定为 single_osc_rgb_processing，并明确 cfa_already_debayered=true；避免把现有逻辑误用于原始 CFA 或马赛克。
  - 对 RGB 边缘分别报告每通道近零率、噪声和候选裁量，最终使用同一矩形；色偏检测触发时标记 radiometric_fallback 和人工复核，不把它声明为共同 coverage 证据。
  - 记录 Stage 1 framing、参考帧、插值方式和裁切前 WCS 是否存在；裁后核验 RGB shape 与 WCS 元数据一致。
- **中期增强**:
  - 为输入模型增加 channel/filter/panel、output_wcs、coverage/weight/mask 与 required_channel_set；所有独立通道先统一 WCS，再一次性求 common processing-safe 矩形。
  - 同时保存每通道 science-union 母版与 mask，以及供现有全图流程使用的共同矩形；按单通道、LRGB、窄带映射和马赛克分别配置产品合同。
  - 若支持 CFA drizzle，生成逐颜色 coverage，实施 Bayer 2×2 与 X-Trans 6×6 相位测试，并禁止 Stage 2 对未去马赛克数据做不对齐裁切。
- **长期架构**:
  - 把输出 WCS 与 framing 决策前移到 Stage 1 注册/叠加，支持 reference、common、threshold-common 和 union+mask；跨通道只重采样一次。
  - 建立 science-safe、processing-safe、display-only 的分支产物与不可逆操作隔离；马赛克默认保持 union、coverage 和非矩形 mask。
  - 以多滤镜错位、CFA 相位、稀疏 drizzle 和多面板合成集回归共同区、颜色边缘、角面积、WCS 和 provenance。


<a id="item-8"></a>
## 8. WCS 与辅助数据传播及科学测量边界

### 研究对象

#### 范围与问题 (`scope_and_question`)

> 研究矩形裁切后怎样原子更新图像尺寸、FITS WCS/CRPIX、mask、DQ、uncertainty、weight、coverage、context、rejection、目录和 PSF 适用域，并规定背景、源检测、孔径测光和 PSF 建模相对不可靠边缘的排除规则。本项限于纯整数切片；若同时旋转、缩放或重投影，还必须执行相应的通量、相关噪声和 WCS 变换，不能套用纯裁切结论。

#### 像素状态分类 (`pixel_state_taxonomy`)

- SCI 有限且可靠：有足够观测贡献、DQ 可接受、权重与不确定度满足任务门禁。
- SCI 有数值但无覆盖：输出填充值可能为 0 或其他常数，weight/coverage/context 表明没有观测。
- 低覆盖或低权重：部分曝光贡献，值可能无明显黑边但误差更大。
- 被拒绝：有几何机会贡献，但有效样本因 DQ、宇宙线或异常值拒绝而不足；需要 rejection 或 context 才能区分。
- 插值或卷积边界：中心像素有效，但其计算支持域接触无效区，适合显示却不一定适合 PSF、反卷积或测量。
- 被 mask 的观测像素：有原始测量但因坏点、饱和、宇宙线或任务规则排除，与 no-data 语义不同。
- 内部有效但 WCS 不一致：像素值可靠，元数据却仍引用裁前网格，会产生错误天球坐标和目录匹配。
- 科学测量安全像素：除自身可靠外，完整测量孔径、背景环、检测滤波器或 PSF 拟合窗口也落在允许区域内。

#### 缺陷成因 (`defect_causes`)

- 裁切只改 SCI 数组却未同步 WCS 参考像素和 NAXIS，会把每个源映射到错误天球位置。
- SCI、mask、ERR、WHT、CTX 或 rejection 采用不同切片、坐标顺序或闭区间约定，形成一到数像素错位。
- FITS 一基像素坐标、NumPy 零基索引、GUI 左上或左下原点混用，导致 CRPIX、目录坐标或上/下裁量符号错误。
- 只保留填零 SCI 而丢弃 coverage 或 DQ，会让下游背景建模和源检测把 no-data 解释为极暗天空。
- 裁切后的畸变 WCS、SIP 或查找表未由支持该模型的工具更新；简单改 CRPIX 可能遗漏辅助畸变表的关联。
- 空间变化 PSF、曝光图、分割图和源目录仍使用裁前坐标域；靠近新边界的星又缺少完整拟合窗口。
- 新边界虽然每个像素有数据，但后续卷积、PSF 拟合、孔径和背景环跨出可靠区，产生边缘偏差。

### 检测与算法

#### 有效性证据与优先级 (`validity_evidence_source`)

- **优先级**:
  - 与 SCI 同源的 DQ/mask、coverage、weight、context、rejection 和 uncertainty
  - 输入 footprint 与纯裁切切片几何，以及每个下游算子的支持域
  - 裁前 WCS 与裁后 WCS 的同一天空位置对应测试和源目录残差
  - 裁后边缘噪声、背景、PSF 和测光诊断
  - 最终像素近零或亮度台阶
- **说明**: AstroDrizzle 的 WHT 是相对权重或有效曝光时间图，CTX 记录哪些输入贡献到各输出像素；JWST 科学产品同时提供 SCI、ERR 和 DQ。此类辅助平面比单一成片值更接近数据生成过程。

#### 所需输入 (`required_inputs`)

- 裁前 SCI、完整 FITS 头和全部 WCS 表示，包括备用 WCS、SIP 或畸变查找表。
- 与 SCI 共网格的 mask、DQ、uncertainty/ERR/variance、weight/WHT、coverage、context/CTX/CON、rejection、有效曝光和分割图。
- 裁切半开区间 x0:x1、y0:y1、原点和轴顺序约定，以及裁切是否伴随翻转、旋转或重投影。
- 源目录、孔径、背景环、PSF/ePSF 模型与空间坐标模型、科学 ROI 及下游最大支持域。

#### 算法步骤 (`algorithm_steps`)

- 冻结裁切定义为同一半开整数切片 [y0:y1, x0:x1]，明确数组轴、GUI 坐标和 FITS 坐标转换；禁止各扩展自行重新检测边界。
- 以一个原子产品容器枚举 SCI 及所有共网格辅助平面，检查裁前 shape、单位、mask 语义和 uncertainty 类型，再用完全相同切片裁切。
- 更新输出尺寸 NAXIS1=x1-x0、NAXIS2=y1-y0。对无翻转的普通二维 WCS，参考像素位置随切片起点平移，即新 CRPIX 等于旧 CRPIX 减相应轴的整数起点；优先调用支持切片的 WCS 库或 Siril 几何操作，而不是手改头。
- 保留 CRVAL、CD/PC/CDELT 和投影语义所代表的天球映射；若有 SIP、畸变查找表、备用 WCS 或轴相关性，由支持该完整模型的库切片并逐项验证。Astropy Cutout2D 明确不支持某些 FITS 畸变查找表。
- 纯裁切不重新估计 SCI、uncertainty、weight 或 coverage，只切片；保持 uncertainty 是标准差、方差还是逆方差的类型和单位。context 位图与 DQ 位域必须原样保留位语义。
- 源目录像素坐标减去裁切起点，天球坐标不变；同步裁切或平移分割图、对象包围盒、背景网格索引和人工 ROI。裁外对象标记为移除，不静默保留旧像素位置。
- 为空间不变且确实适用于全图的 PSF 保留模型本体；空间变化 PSF 的坐标归一化、有效域和节点改到新网格，或重新拟合。ePSF 星样本要求完整 cutout 位于 processing-safe mask 内。
- 根据测量类型建立安全 mask：源检测按滤波核腐蚀，孔径测光要求目标孔径与背景环满足 mask/coverage，PSF 拟合按 fit_shape 或模型支持腐蚀，背景建模单独传 coverage_mask。
- 裁后执行像素平面一致性、WCS 同点、目录匹配、mask 距离和科学测量复检；写入 HISTORY、裁切 provenance，并重新生成需要的校验和。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

> 裁切矩形只定义新数组边界；几何 mask 只说明输入 footprint 是否覆盖。科学可靠 mask 还要合并 DQ、coverage、weight、rejection、uncertainty 有限性和任务允许位。测量安全 mask 再按测量支持域腐蚀：同一个中心像素可对显示有效、对点源检测有效，却因背景环或 PSF cutout 跨边界而不适合测光或 PSF 建模。

#### 内部空洞策略 (`internal_hole_policy`)

- 内部 no-data、坏列、饱和和被拒绝区继续保留在 mask/DQ 中；矩形裁切不能消除它们的语义。
- 源检测与 PSF 拟合使用 mask 排除洞及其支持域；孔径测光报告有效面积、被 mask 比例和质量位，必要时拒绝该源。
- 背景建模把 no-coverage 作为 coverage_mask，把源、坏点等作为普通 mask，不能混为一个填零图。
- 展示副本可插值小洞，但科学母版仍保留原值、原 mask 与插值标记。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

> 在注册/叠加阶段生成 WCS 与辅助平面，在线性图像上完成最终几何裁切，并早于全图背景建模、源检测目录、测光校色、PSF 建模、反卷积、降噪、去星和拉伸。若产品已有可信 WCS，裁切必须当场传播；若像当前项目一样在裁后才 plate solve，也应保留裁前元数据状态并让新解算针对裁后尺寸。科学目录和 PSF 模型应在最终裁切后建立，或明确进行坐标传播和边缘重新筛选。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- **整数矩形切片**: 对现有全图后期管线最稳妥，不新增重采样；所有平面共享同一 x0、y0、x1、y1。
- **最大可靠矩形**: 矩形内部通过可靠 mask，便于背景、卷积和源检测；还要按各测量支持域生成更小的安全中心集合。
- **union加mask**: 保留最大科学视场和不规则覆盖，适合支持 mask 的测量与马赛克；WCS 描述完整输出网格，辅助平面表明哪里可用。
- **旋转或重投影画幅**: 不再是纯裁切，应重新计算输出 WCS、footprint、coverage、weight 和相关误差；最好在注册输出网格阶段一次完成。
- **目录级边缘排除**: 无需为少数测量问题继续裁整图，可在源目录中按 aperture、annulus、fit_shape 和 mask 距离筛选。

#### 安全保护带规则 (`guard_band_rule`)

- 区分图像裁切保护带与科学测量保护带：前者保证矩形像素可靠，后者按每种算子支持域筛选中心。
- 源检测安全距离至少覆盖检测卷积核有效支持；PSF/ePSF 至少覆盖完整 fit_shape 或 cutout；孔径测光同时覆盖源孔径和背景环。
- 存在低覆盖而非硬无效边时，从 coverage/weight 门禁生成 mask 后再做距离变换，不能只量到数组边缘。
- 半开切片向内取整；坐标平移使用实际切片起点，不使用累计比例反算。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

> 目标本体、扩展低表面亮度结构、测光源及比较星、完整孔径、背景环、用于配准或 WCS 校验的参考星、PSF 建模星和背景采样格应作为不同层级 ROI。图像矩形必须保留硬目标 ROI；测量目录再要求每个源的完整支持域处于相应安全 mask。若目标孔径或关键参考星跨越低覆盖边缘，应保留 union+mask 并标记该测量无效，或重新叠加，不能用裁切或填补制造完整观测。

#### 验收与回退 (`acceptance_and_fallback`)

- **科学安全**: SCI、WCS 和全部辅助平面原子同步，可靠 mask 有直接数据来源，测量按各自支持域筛选，provenance 完整且 WCS 同点测试通过。
- **处理安全**: 矩形及必要邻域足以供指定背景、卷积或模型使用；可能没有精密测量所需的完整 DQ/uncertainty，但限制被明确记录。
- **仅展示**: 允许丢弃辅助平面、填零/修补边缘或再次旋转；输出不得覆盖科学母版或用于测光。
- **退化**: WCS 缺失或仅裁后重解、coverage/weight/rejection 未保存、只能由亮度推断 mask，或 PSF 空间域无法传播；下游按限制降级。
- **人工复核**: 畸变查找表不受切片工具支持、备用 WCS 冲突、辅助平面错位、关键 ROI 触边、测光孔径覆盖不足或裁后天球残差超限。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 把 NAXIS 更新视为 WCS 已更新；实际 CRPIX 或畸变辅助表仍指向裁前像素网格。
- 直接套用 CRPIX 平移公式却混淆 FITS 一基坐标、NumPy 零基索引、y 轴方向或已经发生的翻转。
- Astropy Cutout2D 能更新普通 WCS，但官方明确警告不处理某些 FITS 畸变查找表；无条件依赖会产生看似有效的错误 WCS。
- 只裁 SCI，WHT/ERR/DQ/CTX 保持原尺寸；程序若广播、截断或按旧坐标读取，会产生隐蔽科学错误。
- 把 WHT 当作统一物理量；它可能表示曝光权重、误差权重或逆方差，裁切不改变类型，但门禁必须知道定义。
- 把 CTX/CON 当 coverage 数值直接相加；它通常是贡献图像位图，需要按位解释，且输入多于位宽时可能有额外平面。
- mask 后仍使用未校正的孔径面积或背景环，造成通量和背景偏差；简单忽略坏像素不等于完成缺失面积校正。
- 空间变化 PSF 在新坐标网格沿用旧归一化坐标，或选择 fit_shape 跨越低覆盖区，会造成位置相关系统误差。
- 重新 plate solve 可以得到裁后 WCS，但不能恢复丢失的 DQ、weight、uncertainty 或输入贡献 provenance。
- 为保证 PSF 窗口完整而大幅裁整图会无谓删除科学区域；通常应保留图像并在目录级排除边缘源。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

> 若 SCI 仍是 CFA，SCI 与 DQ、mask、weight 必须在同一传感器像素格切片，并保持 Bayer 2×2 或 X-Trans 6×6 相位以及 ROWORDER/XBAYROFF/YBAYROFF；去马赛克或 CFA drizzle 后的 RGB 各通道、weight 与 mask 使用同一输出 WCS。多滤镜产品的 WCS 同步只是必要条件，科学测量还必须使用各滤镜自己的可靠 mask 与 uncertainty；共同目录坐标可共享天球坐标，但像素有效域按通道判断。

#### WCS 与辅助平面传播 (`wcs_and_auxiliary_propagation`)

- **WCS与尺寸**: 更新实际数组尺寸、NAXIS、CRPIX 与像素边界；纯裁切保持同一天空映射和尺度/方向。同步处理主 WCS、备用 WCS、SIP 和畸变信息，遇到不支持的查找表则重解或用专门工具。
- **像素平面**: SCI、mask、DQ、uncertainty/ERR、variance/inverse-variance、WHT、coverage、CTX/CON、rejection、有效曝光、分割图和空间图使用同一切片；纯裁切不改单位与数值定义。
- **目录与区域**: 像素坐标、包围盒、分割标签、孔径和 ROI 减裁切起点；天球坐标保持不变。裁外对象明确移除，部分裁切对象设置质量位。
- **PSF**: 固定 PSF 图样本体未必需要裁切，但其适用域和坐标元数据必须更新；空间变化 PSF、PSF 网格与曝光相关模型需变换坐标或重新拟合，边缘星必须有完整可靠 cutout。
- **元数据与来源**: 更新 HISTORY、裁切矩形、软件版本、输入产品关联和校验和；保留 weight 与 context 的语义、DQ 位定义和 uncertainty 类型。

#### 下游影响 (`downstream_impact`)

> 正确传播后，背景建模可用 coverage_mask 排除马赛克空白，源检测可用 mask 避免在填充值台阶上产生伪源，孔径和 PSF 测光能用 error 与 DQ 计算或筛选结果，目录天球位置保持一致。若传播失败，背景面会向零边缘弯曲，检测卷积会在边界制造结构，测光孔径与背景环会缺面积，PSF/ePSF 会被截断或采用错误空间坐标，光度校色与跨滤镜匹配随之偏移。反卷积和去星还会把边界错误扩散到内部。过度图像级裁切则减少参考星和弱信号；保留 union+mask 并在目录级排除通常更保真。

#### 质量验证 (`qa_and_validation`)

- 断言每个共网格扩展与 SCI shape 完全相等，输出 NAXIS 匹配数组；纯裁切时逐像素比较输出与输入 [y0:y1,x0:x1]。
- 在裁后四角、中心、ROI、随机点和共同恒星上比较裁前 (x+x0,y+y0) 与裁后 (x,y) 的世界坐标，并做世界到像素往返；同时检查像素尺度、方向和边界天球多边形。
- 逐位统计裁前候选区与裁后 DQ、context 和 rejection，比较 WHT、coverage、ERR 分位数；确认 mask 语义和 uncertainty 类型没有变化。
- 将 SCI、WHT/coverage、DQ/mask、context 贡献数、processing-safe mask、源孔径、背景环与 PSF cutout 叠图检查。
- 对源检测比较边缘伪源率；对裁前同一区域与裁后目录比较源数、质心、孔径通量、误差、PSF/FWHM 和质量位。
- 构造平移裁切、四个边界、奇偶尺寸、SIP、畸变查找表、多个备用 WCS、多扩展 shape 错位、内部洞和部分孔径的回归数据。
- 重新打开最终 FITS，验证所有扩展、WCS、单位、HISTORY 和校验和，避免仅在内存对象正确。

#### 裁切溯源 (`crop_provenance`)

> 记录输入文件及校验标识、原始和最终尺寸、精确半开切片与坐标约定、累计四边裁量、裁前后主/备用 WCS 摘要、畸变类型、全部辅助扩展及其语义、DQ 允许位、weight/uncertainty 类型、coverage 门禁、各科学算子支持域、目录移除或质量位统计、WCS/测光 QA 结果、软件与库版本、是否额外重采样、人工覆盖和产品安全等级。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- FITS 官方标准与 WCS 论文把 CRPIX 定义为参考像素位置、CRVAL 为其世界坐标；子图必须保持像素到世界的映射而不是只复制旧头。
- Astropy NDData/CCDData 把 data、mask、uncertainty、flags 和 WCS 作为关联对象；官方文档说明切片会相应切片 mask 和 WCS，CCDData 可将 MASK、UNCERT、flags 与 PSF 写入 FITS 扩展。
- Astropy Cutout2D 可返回更新后的 WCS 并提供原图与子图像素坐标转换，但明确警告当前不处理 FITS WCS distortion paper 的查找表。
- AstroDrizzle 最终产品包含 SCI、WHT 和 CTX；官方定义 WHT 为相对权重或有效曝光时间图，CTX 记录哪些输入图像贡献到各像素。
- JWST 科学数据通常同时提供 SCI、ERR 与 DQ，部分重采样产品还提供 WHT/CON 或 WMAP，证明科学图像应与质量和误差平面一同传播。
- Photutils Background2D 明确区分 no-coverage 的 coverage_mask 与源/坏像素 mask；源检测和 PSF 测光接受 mask，非有限值会被排除。
- Photutils extract_stars 不提取 cutout 部分或全部越出输入图像的星，直接支持 ePSF 星必须远离边界至少半个 cutout 的规则；孔径工具支持 mask、误差和边界重叠。
- Siril 自 1.0 起官方声明 rotate、mirror 和 crop 会更新 WCS；这支持当前项目调用 Siril crop，但仍需版本锁定与裁后验证。
- 按各科学算子构造不同安全 mask、在目录级排除边缘源以及五级产品验收是本调研的工程综合。

#### 直接证据来源 (`evidence_sources`)

- **标题**: FITS 官方标准与 WCS 规范索引 | **网址**: https://fits.gsfc.nasa.gov/fits_wcs.html | **来源类型**: 美国航天局 FITS 官方标准入口 | **日期**: 访问于 2026-08-04
- **标题**: Astropy CCDData：data、mask、uncertainty、flags、WCS 与 PSF | **网址**: https://docs.astropy.org/en/stable/api/astropy.nddata.CCDData.html | **来源类型**: 官方软件 API 文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: Astropy NDData：关联平面切片与不确定度传播 | **网址**: https://docs.astropy.org/en/stable/nddata/index.html | **来源类型**: 官方软件文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: Astropy Cutout2D：WCS 更新与畸变查找表限制 | **网址**: https://docs.astropy.org/en/stable/api/astropy.nddata.utils.Cutout2D.html | **来源类型**: 官方软件 API 文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: HST AstroDrizzle 管线产品：SCI、WHT 与 CTX | **网址**: https://hst-docs.stsci.edu/drizzpac/chapter-5-drizzlepac-software-package/5-3-astrodrizzle-in-the-pipeline | **来源类型**: 空间望远镜科学研究所官方文档 | **日期**: 访问于 2026-08-04
- **标题**: JWST 科学数据概览：SCI、ERR、DQ、WHT 与 context | **网址**: https://jwst-docs.stsci.edu/accessing-jwst-data/jwst-science-data-overview | **来源类型**: 空间望远镜科学研究所官方文档 | **日期**: 访问于 2026-08-04
- **标题**: Photutils Background2D：coverage_mask 与普通 mask | **网址**: https://photutils.readthedocs.io/en/stable/api/photutils.background.Background2D.html | **来源类型**: 官方软件 API 文档 | **日期**: 版本 3.0.0；访问于 2026-08-04
- **标题**: Photutils 源检测：mask 像素不参与检测 | **网址**: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.detect_sources.html | **来源类型**: 官方软件 API 文档 | **日期**: 稳定版文档；访问于 2026-08-04
- **标题**: Photutils PSF 测光：data、error、mask 与非有限值 | **网址**: https://photutils.readthedocs.io/en/stable/user_guide/psf.html | **来源类型**: 官方软件文档 | **日期**: 版本 3.0.0；访问于 2026-08-04
- **标题**: Photutils extract_stars：边界不完整 cutout 的排除 | **网址**: https://photutils.readthedocs.io/en/latest/api/photutils.psf.extract_stars.html | **来源类型**: 官方软件 API 文档 | **日期**: 开发版文档；访问于 2026-08-04
- **标题**: Photutils 孔径测光：mask、误差与孔径边界 | **网址**: https://photutils.readthedocs.io/en/stable/user_guide/aperture.html | **来源类型**: 官方软件文档 | **日期**: 版本 3.0.0；访问于 2026-08-04
- **标题**: Siril 1.0：裁切、旋转和镜像后的 WCS 更新 | **网址**: https://siril.org/download/2021-11-20-siril-1.0.0-rc1/ | **来源类型**: 官方版本说明 | **日期**: 2021-11-20；访问于 2026-08-04

#### 当前项目适配情况 (`current_project_fit`)

- 当前 Stage 2 通过 Siril crop 对当前图像做整数矩形裁切，并在 stage2_crop_report.json 记录每次 x、y、width、height、移除四边和累计裁量；这些信息足以成为 WCS 与目录平移 provenance 的基础。
- 项目只把 stage2_corrected.fit 作为统一输出，没有生成或同步 mask、DQ、uncertainty、weight、coverage、context、rejection、分割图或 PSF 辅助产品，因此当前只能达到单 SCI 的处理路径。
- Stage 2 主要依赖 RGB 像素统计与 edge_black_ratio，无法区分填零 no-data、低权重、被拒绝样本和真实暗背景，也无法为测光或 PSF 建模产生可靠 mask。
- 当前 stage2_guard_band_pixels=3 是边缘检测保护像素，不代表背景网格、卷积、去星、反卷积或 PSF fit_shape 的科学支持域。
- Stage 4 才执行 platesolve，因此常规路径的新 WCS 针对裁后 stage2_corrected 图建立，规避了新解算 WCS 的裁切传播；但若用户输入的叠加图已有 WCS，项目没有显式保存裁前后同点 QA。
- Siril 官方自 1.0 起声明 crop 会更新 WCS，说明当前命令有合理底层支持；项目仍应锁定实际 Siril 版本并验证保存后的 FITS 头，尤其是 SIP 或其他畸变。
- 后续阶段主要读取单一图像而非数据、mask、不确定度组合，边界若仍不可靠只能通过整图裁切或 degraded 状态控制，不能在源目录或算法级精确排除。
- 从 stage2_corrected.fit 续跑时不会重做裁切；报告和 FITS 内必须自包含裁切与 WCS provenance，否则无法可靠恢复辅助坐标关系。

#### 实施建议 (`implementation_recommendations`)

- **最高优先级最小改动**:
  - 扩展 stage2_crop_report.json：记录明确的半开切片、坐标原点/轴方向、裁前后 FITS 尺寸、WCS 是否存在、CRPIX 摘要、Siril 版本及裁后 WCS 同点检查结果。
  - 保存 stage2_corrected.fit 后重新读取头和 shape；若裁前有 WCS，在中心、四角和随机点验证裁前偏移坐标与裁后坐标的天空位置一致，失败则标记 manual-review/degraded。
  - 把固定 3px 明确命名为 edge-detection guard，不宣称 science-safe；后续背景、卷积和模型分别使用自己的安全 mask 或边缘排除宽度。
- **中期增强**:
  - 引入最小多扩展或并行 FITS 产品：SCI、VALIDMASK、COVERAGE/WEIGHT 和可选 REJECTION/UNCERTAINTY；Stage 2 原子切片并验证全部 shape。
  - 在后续背景模块传入 coverage_mask，在源检测、测光和 PSF 模块按 fit_shape、孔径及背景环距离筛选；用目录质量位代替继续裁整图。
  - 记录并传播目录像素坐标、ROI、分割图和空间 PSF 坐标域；为普通 WCS、SIP 和不支持的查找表建立不同处理/回退路径。
- **长期架构**:
  - 将图像改为包含 SCI、WCS、mask、uncertainty、weight、coverage、context/rejection 和 provenance 的原子数据产品，任何几何操作由单一切片接口完成。
  - 让 Stage 1 从注册/叠加直接产出 footprint 与覆盖辅助平面；Stage 2 输出 science-union+mask 和 processing-safe 矩形，而展示副本单独生成。
  - 建立合成与真实回归矩阵，覆盖坐标基准、四边裁切、多扩展错位、SIP/查找表、孔径触边、PSF cutout、内部洞、WCS 往返和光度不变量。


<a id="item-9"></a>
## 9. 主流软件与天文数据管线实践

### 研究对象

#### 范围与问题 (`scope_and_question`)

- 核验 Siril、PixInsight、Astro Pixel Processor、DeepSkyStacker、Astropy/reproject、DrizzlePac 与 Montage 在配准、重投影、叠加和裁切时如何定义输出画幅、识别无覆盖或低可靠像素，以及传播 WCS 与质量辅助平面。
- 重点回答这些工具是否自动裁除场旋黑角、是否能选择共同区/参考区/并集/自定义区，以及用户应依据最终像素亮度还是 footprint、weight、context、DQ、area 等证据。
- 不把商业软件界面中未公开的内部阈值推断成事实；也不把用于美学构图的人工裁切等同于支持测光或定量分析的科学裁切。

#### 像素状态分类 (`pixel_state_taxonomy`)

- Siril、APP、DSS 等消费级处理软件常把配准画幅外区域表现为黑边或未共同覆盖区，但黑色数值本身仍可能与真实暗天空混淆。
- reproject 的 footprint=0 表示输出像素不落在原图 footprint 内；精确球面多边形重投影还能给出 0 到 1 的分数覆盖。
- AstroDrizzle 的 WHT 表示输出像素权重，CTX/CON 表示哪些输入曝光贡献；DQ 决定哪些输入像素可参与，这些状态与 SCI 亮度分离。
- Montage 的 area 图记录投影/合成到输出像素的天空面积，可作为权重及输出保真验证依据；mAdd 的 count 模式可生成参与图像计数。
- PixInsight ImageIntegration 的 low/high rejection map 表示被拒绝的样本，不等同于无覆盖；StarAlignment 生成的 mask 可用于检查马赛克注册边界。
- 内部孔洞、低权重、插值核边界、真实暗背景和人工填充需要保持为不同状态，不能统一编码成零值后再凭亮度裁切。

#### 缺陷成因 (`defect_causes`)

- 场旋、dither、平移、不同相机角度、尺度和畸变校正使配准图像的 footprint 不再共享同一个轴对齐矩形，从而形成黑角、斜边或贡献数下降的外圈。
- union/max/mosaic 画幅保留全部输入视场，因此天然包含部分覆盖区；common/min/intersection 画幅牺牲视场换取所有输入共有的区域。
- Drizzle 的 pixfrac、输出 scale、核、输入帧数与 dither 覆盖不足可产生零权重孔洞；过严 DQ 或异常值拒绝也会减少有效贡献。
- Lanczos、双三次等重采样可能在星点和硬边附近产生 ringing；Siril 和 APP 都提供抑制过冲/欠冲或 clamping 的相关选项。
- 马赛克各面板的背景、归一化和 PSF 不一致会产生接缝或色边；这类问题需归一化、背景匹配或 blending，不能仅靠裁去最外圈保证解决。

### 检测与算法

#### 所需输入 (`required_inputs`)

- 配准变换或每幅输入的 WCS/shape、输出 WCS/shape、叠加 SCI 图。
- 可用时读取 footprint、DQ/mask、WHT/area、CTX/CON、参与帧数和 rejection map。
- 用户选择的 framing 模式、科学 ROI/参考帧、通道关系、插值或 drizzle 核及其参数。
- 裁切前后的 FITS header、WCS、像素尺度、方向、目标和参考星坐标。

#### 算法步骤 (`algorithm_steps`)

- 一、先选择输出几何：共同区适合稳定全图处理，参考画幅适合固定构图，并集/马赛克适合保留最大天区，自定义 ROI 适合明确科学目标。
- 二、把所有输入 footprint 投影到输出 WCS；同时投影 DQ/mask/weight，形成 geometric coverage、有效贡献数和可靠权重。
- 三、按照任务定义把零覆盖、低覆盖、被拒绝、插值支持不足与真实暗背景分开；不以 SCI=0 单独决定有效性。
- 四、共同区产品可直接在达到所有必需输入/通道门限的掩膜中求矩形；union 产品保留非矩形 footprint 和辅助平面，不强制把所有空角裁掉。
- 五、根据后续插值、卷积、反卷积或模型支持域腐蚀可靠掩膜，再应用 ROI、参考星和视场损失约束。
- 六、同步裁切 SCI、mask、weight、coverage、context、rejection 和 uncertainty，并更新或重建 WCS。
- 七、用 footprint/WHT/area/CTX、拒绝图、WCS 往返和视觉弱拉伸联合验收；生成科学产品与展示产品的不同接受等级。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

- Siril min/max、DSS Intersection/Mosaic、APP Reference/Full/Crop、PixInsight Register/Union 与 ROI 主要定义 geometric framing。
- reproject footprint、AstroDrizzle WHT/CTX/DQ、Montage area/count 才能进一步定义数值可靠性；有几何覆盖不代表权重、贡献数或插值支持足够。
- 建议软件适配层统一输出 geometric_mask、reliable_mask 和 processing_safe_mask，避免把某一软件的‘已裁切’误认为所有像素均为科学可靠。

#### 内部空洞策略 (`internal_hole_policy`)

- Drizzle/reproject/Montage 产品应通过 footprint、WHT、CTX 或 area 保留内部孔洞；矩形裁切只能消除与外边连通的无效区。
- Siril 文档指出某些 drizzle 核或输入帧不足会形成零像素斑驳，可改用 square/Gaussian 或增加帧数；这优先于裁掉含孔洞的整个大矩形。
- 少量坏点或芯片间隙继续以 mask 传播；展示副本可修补，但不得把修补后的像素升级为科学观测。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

- 几何画幅应在配准变换确定后选择；若能在叠加时直接输出 common/reference/ROI，可避免先生成大面积黑边再依亮度猜测。
- 最终可靠裁切应在叠加后、背景建模和校色前完成，并保留叠加产生的 coverage/weight/rejection；这与 Siril 常见先裁配准边再做后期的实践一致。
- 马赛克或科学产品可保留 union 画幅及 mask/weight；用于卷积、反卷积、去星或神经网络时另保留工作 halo，最终展示阶段再生成更紧的 display crop。
- 多通道合成前必须先统一 WCS、shape 和共同有效支持；APP 官方论坛也强调仅靠尺寸或事后裁切不能替代通道重新注册。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- Siril：current、min、max、cog；min 是全序列共同区，max 是包围所有输入的画幅，cog 以重心改善构图。
- DeepSkyStacker：Normal 由参考帧定框，Mosaic 包含全部 light，Intersection 由所有帧交集定框，Custom Rectangle 在用户矩形内处理。
- Astro Pixel Processor：Full 扩展到包含输入偏移后的全部视场，Reference 保证与参考画幅一致，Crop 可在集成时选定参考画幅中的有限区域；多通道/马赛克需共同注册。
- PixInsight：StarAlignment 可做 match/union mosaic 并生成 mask，DynamicCrop/预览 ROI 提供人工或明确区域裁切；公开资料未显示一个基于亮度自动认定科学可靠边缘的统一工具。
- reproject：find_optimal_celestial_wcs 计算容纳全部输入的 WCS/shape，auto_rotate 可最小化最终包围面积；reproject_and_coadd 返回 mosaic 与 footprint。
- DrizzlePac：默认输出为容纳完整 dithered field 的最小 WCS 画幅，也可由 final_refimage、final_outnx/outny、中心、旋转和尺度完整指定。
- Montage：header template 明确定义输出天区和投影，mMakeHdr 可从输入求包围头，mAdd 可按模板 exact-size 或收缩到实际数据范围；area/count 继续表达覆盖。

#### 科学目标 ROI 约束 (`science_roi_constraint`)

- 选择 common/intersection/min 不能只追求无黑边；若弱信号、比较星、天空环或马赛克面板落在低覆盖边缘，应显式选择 reference/union+mask 或用户 ROI。
- APP Crop、DSS Custom Rectangle、PixInsight ROI/preview 和 DrizzlePac custom WCS 都说明 ROI 是用户或任务约束，而非自动从亮度推断。
- 多通道产品需在所有必需通道共同 WCS/shape 上保护目标；APP 官方管理员明确指出不同 full-composition 尺寸意味着通道未必在同一坐标框架内。
- 科学 ROI 内若存在 WHT=0、footprint=0 或 area=0，应优先修复叠加/采样或保留 mask，而不是用填充像素消除警报。

#### 验收与回退 (`acceptance_and_fallback`)

- science-safe：footprint/DQ/weight/context 或 area 等高优先证据齐全，ROI 和辅助平面完整，WCS 验证通过。
- processing-safe：适合背景、校色和常规后期，但边缘 coverage 或误差模型不足以支持定量测量；保留 mask/weight。
- display-only：使用 union 空角填充、修补或美学构图，必须与科学 FITS 分开。
- degraded：只能使用最终亮度/色偏推断，或软件版本行为未能核验；允许保守继续但输出残留风险。
- manual-review：common 与 ROI 冲突、马赛克 footprint 非矩形、多通道覆盖不一致、内部孔洞或候选间损失差异很大。
- 回退顺序：调整 framing/输出 WCS或重叠加 → 保留大画幅并传播 mask/weight → 跳过高风险下游 → 独立 display crop。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 把 SCI=0 或近黑当 no-data 会误裁真实暗星云、低表面亮度外延和经过背景扣除的负值。
- 共同区受单个偏移异常帧控制，可能比基于最低贡献数或权重的可靠核心小很多。
- union/max/mosaic 输出的空角被填黑后，若丢失 footprint/weight，后续背景统计会被污染。
- 只检查外边条带会漏掉 drizzle holes、坏列、芯片缝和不规则马赛克内部缺口。
- 只看 rejection map 会把成功剔除宇宙线与无覆盖混淆；只看 weight 非零又可能接受极低权重边缘。
- 对 CFA 在奇数坐标裁切或任意角旋转会改变采样相位；不同通道分别自动裁切会破坏共同坐标。
- 商业软件教程、论坛和当前版本可能不同；没有版本号与产物检查时，不能把界面行为当长期契约。

### 数据完整性与影响

#### CFA 与通道约束 (`cfa_and_channel_constraints`)

- Siril 官方文档明确：Bayer 裁切按 2×2 边界对齐，X-Trans 按 6×6；任意角旋转应在 debayer 或 Bayer drizzle 后进行。
- APP 多通道/多滤镜工作流要求共同注册并生成相同画幅；Reference composition 可保持跨通道一致。
- PixInsight/WBPP 与 Drizzle 多通道产品也应共享输出 WCS/shape，并在全部必需通道的可靠掩膜交集中验收。
- DSS 可对最终 RGB 通道执行对齐，但这不能恢复已被各通道不同裁切移除的天区。

#### 下游影响 (`downstream_impact`)

- 背景建模：空角、低权重和面板接缝会扭曲背景位置与尺度；过度 common crop 又会减少背景采样区域。
- 校色和通道合成：不同画幅或 WCS 会造成颜色错位；边缘参考星数量和空间分布下降会降低稳健性。
- 反卷积、降噪和去星：零填充或低支持边缘会产生 ringing、接缝和模型伪影，需提前裁切或保留工作 halo。
- 拉伸：低权重噪声、黑边和彩边被放大；展示产品可以更积极裁边，但不得覆盖科学产品。
- 测光与源检测：必须用 WHT/area/mask 限制有效区域；被拒绝样本、低覆盖和无覆盖需要不同误差处理。
- 马赛克：背景匹配、LNC/MBB 或 frame adaptation 解决接缝，不应把所有问题归结为外边裁切。

#### 质量验证 (`qa_and_validation`)

- Siril：对比 framing=current/min/max/cog 的 shape，检查裁切后黑角、星边 ringing、CFA 相位和 WCS。
- PixInsight：检查 StarAlignment mask、ImageIntegration low/high rejection map，并以弱拉伸确认没有把真实结构当拒绝或边缘伪影。
- APP：检查 Registered viewer、跨通道 shape/WCS、normalization map、LNC/MBB 接缝与 Crop/Reference/Full 画幅。
- DSS：分别验证 Normal/Mosaic/Intersection/Custom Rectangle 的实际输出尺寸和参考帧关系；避免在 drizzle/超像素设置变化后沿用旧注册。
- reproject：可视化每个 footprint 与累计 footprint，检查 exact/interp 边界差异、输出 WCS角点和 auto_rotate 结果。
- DrizzlePac：检查 SCI/WHT/CTX、DQ bits、零权重 holes、输出 WCS/scale/rotation 和 ROI 内权重均匀性。
- Montage：检查 mosaic、area/count 图、header template、mCoverageCheck/mSubset 选择和背景修正前后接缝。
- 跨工具统一做 pixel→world→pixel 往返、辅助平面 shape/origin 一致性、ROI/参考星保留和裁切前后面积/角视场报告。

#### 裁切溯源 (`crop_provenance`)

- 记录软件名、精确版本、framing/composition/result 模式、参考帧或 header template、输出 WCS/shape。
- 记录输入 footprint/DQ/weight/context/area/rejection 来源、阈值、插值/drizzle 核、pixfrac、scale 与 DQ bits。
- 记录候选和最终矩形相对原图坐标、每边/面积/角视场损失、ROI 与通道约束、接受等级和人工覆盖。
- 为 SCI 与所有辅助平面记录校验和，确保续跑不会只加载图像而丢失边界证据。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- Siril 官方文档：seqapplyreg 提供 current/min/max/cog；min 裁到全序列共同区，max 计算包围所有图像的画幅；旋转不裁切时缺失处填黑，bicubic/Lanczos 默认 clamping；CFA 裁切遵守 2×2 或 6×6 相位。
- PixInsight 第一方示例/论坛：StarAlignment 可用 Register/Union-mosaic 并生成 mask 检查注册，DynamicCrop 用于移除注册伪影；ImageIntegration 输出拒绝图用于检查异常值拒绝。未找到公开的统一自动亮度裁边规范。
- APP 第一方管理员说明：Full 会扩展画幅以包含全部输入偏移，Reference 保证最终通道共用参考画幅，6)Integrate 的 Crop 可只处理参考画幅中的选择区；多通道马赛克应共同注册，LNC/MBB 用于归一化和接缝。
- DSS 官方开源代码：Normal 由参考 light 定框，Mosaic 包含全部 light，Intersection 由所有帧交集定框，Custom Rectangle 是 Normal 加用户矩形；官方 FAQ 还警告更换 super-pixel 模式后需重新注册。
- Astropy/reproject 官方文档：所有重投影函数返回数据和 footprint；exact 方法可返回分数重叠；find_optimal_celestial_wcs 计算容纳所有输入的 WCS/shape并可 auto_rotate 减小包围面积；coadd 同时返回累计 footprint。
- DrizzlePac 官方文档：AstroDrizzle 将 DQ 允许位、输入权重、pixfrac、kernel 和输出 WCS结合，正式产品包含 SCI/WHT/CTX；默认输出可容纳完整 dithered field，也能完整指定自定义 WCS。
- Montage 官方文档：mProject 生成重投影图和 area 图，mAdd 根据 header template 合成并生成输出 area，可用 count 统计重叠图像数；mCoverageCheck/mSubset 按天区 footprint 选择输入。
- 综合推断：这些实践共同支持‘先用几何/权重证据定义可靠区，再按任务 ROI 选矩形；亮度扫描仅作缺少辅助平面时 fallback’，但这一统一分级是本调研的工程归纳，不是某个软件的原生命令。

#### 直接证据来源 (`evidence_sources`)

- Siril《Drizzle 与 seqapplyreg framing》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- Siril《几何：Rotate & Crop 与插值 clamping》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/processing/geometry.html
- Siril《主界面：CFA 裁切约束》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- PixInsight《Building large mosaics with StarAlignment》，第一方论坛示例，2010 年，https://pixinsight.com/forum/index.php?threads/building-large-mosaics-with-staralignment.1953/
- PixInsight《Messier 31 H-alpha Processing Notes》，官方处理说明，访问日期 2026-08-04，https://www.pixinsight.com/examples/M31-Ha/
- Astro Pixel Processor《RGB combination and same-size requirements》，第一方管理员答复，2017-08-10 至 2019-10-10，https://www.astropixelprocessor.com/community/faq/rgb-combination-and-same-size-requirements/
- Astro Pixel Processor《Cropping a mosaic》，第一方管理员答复，2024-02-15，https://www.astropixelprocessor.com/community/main-forum/cropping-a-mosaic/
- Astro Pixel Processor《Bayer drizzle》，第一方管理员答复，2021-03-20，https://www.astropixelprocessor.com/community/tutorials-workflows/bayer-drizzle/
- DeepSkyStacker《ResultParameters.cpp》，官方开源实现，访问日期 2026-08-04，https://github.com/deepskystacker/DSS/blob/master/DeepSkyStacker/ResultParameters.cpp
- DeepSkyStacker《FAQ》，官方网站，访问日期 2026-08-04，https://deepskystacker.free.fr/english/faq.htm
- Astropy/reproject《Footprint arrays》，官方文档，更新于 2026-06-25，https://reproject.readthedocs.io/en/stable/footprints.html
- Astropy/reproject《Combining images into mosaics》，官方文档，访问日期 2026-08-04，https://reproject.readthedocs.io/en/stable/mosaicking.html
- Astropy/reproject《reproject_and_coadd》，官方 API，访问日期 2026-08-04，https://reproject.readthedocs.io/en/stable/api/reproject.mosaicking.reproject_and_coadd.html
- Astropy《CCDData Class》，官方文档，访问日期 2026-08-04，https://docs.astropy.org/en/stable/nddata/ccddata.html
- STScI《DrizzlePac Handbook 3.0》，官方手册，2025-01，https://hst-docs.stsci.edu/drizzpac
- DrizzlePac《AstroDrizzle API》，官方文档，访问日期 2026-08-04，https://drizzlepac.readthedocs.io/en/latest/_modules/drizzlepac/astrodrizzle.html
- DrizzlePac《Single-visit Mosaic Processing》，官方文档，访问日期 2026-08-04，https://drizzlepac.readthedocs.io/en/latest/mast_data_products/svm_processing.html
- Montage《Toolkit Components》，IPAC 官方文档，访问日期 2026-08-04，https://irsa.ipac.caltech.edu/Montage/docs/components.html
- Montage《Mosaic Notebook》，官方示例，访问日期 2026-08-04，https://montage.ipac.caltech.edu/MontageNotebooks/Mosaic.html
- Montage《M101 Mosaic Tutorial》，IPAC 官方教程，访问日期 2026-08-04，https://irsa.ipac.caltech.edu/Montage/docs/m101tutorial.html

#### 实施建议 (`implementation_recommendations`)

- P0：把 Stage 1 使用的 Siril framing 作为显式配置和报告字段，至少区分 reference/current、common/min 和 union/max；不要依赖版本默认值。
- P0：在现有 Stage 2 报告中加入 evidence_source、framing_mode、输入软件版本、WCS/shape、面积与角视场损失、接受等级；像素扫描明确标为 fallback。
- P0：若输入 FITS 已包含 mask/weight/coverage 类扩展或旁车，优先读取并与 SCI 同窗裁切；缺失时才走现有 near-black/chroma 检测。
- P1：对原始 Light 路径从已注册序列生成 coverage_count 或有效曝光权重图，Stage 2 先按 coverage 定义 reliable mask，再使用亮度/色偏诊断插值和通道伪影。
- P1：实现 common/reference/union 三种产物策略；common 求可靠核心，reference 保持固定画幅并带 mask，union 用于马赛克或最大视场且禁止将黑填充当真实背景。
- P1：为 SCI、mask、coverage/weight、WCS 和 ROI 建立同一裁切事务与 shape 校验；resume checkpoint 需携带这些旁车和校验和。
- P1：按插值核、配准残差和后续算子 halo 计算 guard band，替代固定像素；CFA 输入强制周期对齐或先拒绝进入 Stage 2。
- P2：借鉴 DrizzlePac/Montege 的质量产品，保存 coverage、weight/context、rejection 与边界 QA 图；建立 science-safe、processing-safe、display-only、degraded、manual-review 五级输出。
- P2：建立跨软件回归集，验证 Siril framing、DSS/APP/PixInsight 导入产物、Drizzle/reproject/Montage 权重图、WCS 往返和暗星云误判。


<a id="item-10"></a>
## 10. 当前项目 Stage 2 差距与实施建议

### 研究对象

#### 范围与问题 (`scope_and_question`)

- 逐项审计当前 Stage 2 的中心背景估计、四边近黑/亮度/色偏扫描、场旋黑角处理、edge_black_ratio 迭代、固定保护带、彩边裁切、裁切报告和 stage2_corrected_resume 契约。
- 把实现与 footprint、DQ/mask、coverage/weight、ROI、WCS 和辅助平面传播等调研结论比较，给出不改变 Stage 1-10 顺序前提下的分层改造方案。
- 本项只给出实施建议，不直接修改 pipeline 算法、阶段顺序或用户文档；所有建议需在真实 Seestar 数据回归集上标定后才能成为默认值。

#### 像素状态分类 (`pixel_state_taxonomy`)

- 当前实现把 NaN、正负无穷在图像转换时归零，因此随后会与黑填充、截黑像素一起进入近黑检测，丢失 non-finite 独立状态。
- near-black：由中心背景推导阈值或 image_metrics._dark_clip_threshold 判定，可能代表配准画幅外填充，也可能是真实暗背景。
- 暗/亮台阶：按整行或整列中位数及 81 像素滚动中位数识别，可能代表叠加边界，也可能是真实大尺度天体/渐晕。
- 彩色伪影：按 RGB 相对灰度的中位色偏及红/蓝单通道 excess 判断，但没有区分通道 footprint 不一致、色差、真实发射/反射结构。
- 低覆盖、被拒绝、插值、内部孔洞、坏列和真实暗结构目前没有独立标签；最终只有裁切后的 RGB 图和 ok/degraded 状态。

#### 缺陷成因 (`defect_causes`)

- Stage 1 的 register -2pass 与 seqapplyreg 会施加几何变换；当前没有显式 -framing 参数，输出边缘可能由参考画幅、平移、旋转和插值填充产生。
- 输入也可能是用户提供的已叠加母版，此时黑边来源、叠加模式、通道覆盖和 WCS 质量未知。
- 四边亮度台阶、红蓝边缘可能来自场旋低覆盖、插值 ringing 或通道错位，也可能来自渐晕、星云、暗云、光害梯度和真实颜色。
- fixed 3px guard 每次对已检出边继续追加；多轮自适应会重复消耗边缘，但其宽度并未由实际插值核或后续算子支持域推导。

### 检测与算法

#### 所需输入 (`required_inputs`)

- 现状：Siril 当前全分辨率像素、图像 shape、PipelineConfig 和图像特征测量结果。
- P0 增补：原图尺寸/WCS、Stage 1 framing、总裁切预算、目标位置或保守中心保护区、每次候选前后指标。
- P1 增补：每帧配准变换或 footprint、coverage_count/weight、DQ/rejection、目标/弱信号 ROI、参考星和背景可用区。
- 续跑契约：stage2_corrected.fit 与 crop report、mask/coverage、WCS 和配置指纹的同一检查点清单。

#### 算法步骤 (`algorithm_steps`)

- 现状一：读取全分辨率像素并转换为 3 通道 float；单通道复制为 RGB，非有限值归零，大于 2 的数据按全图最大值归一化。
- 现状二：以中心 50%估计背景和色偏，计算整行/列近黑比例、中位亮度、色偏与滚动亮度台阶，得到 bad_rows/bad_cols。
- 现状三：从四边扫描至连续稳定正常段，随后用四个 5×5 黑角循环对左右/上下同时增量，再为每个非零边加固定 guard。
- 现状四：执行 Siril crop 并累计 crop_report；若 edge_black_ratio 超标，则在已裁图上重新检测，按单轮 3.5%上限继续裁，最多三轮。
- 现状五：每轮裁后复测；改善不足 0.003 时标 degraded 并停止，但不会撤销刚执行的候选裁切。最终仍超目标也 degraded。
- 现状六：仅在状态仍为 ok 时检查彩边并最多再裁每侧约 2.5%；保存 stage2_corrected.fit 和 stage2_crop_report.json。
- 建议一：从原始 Stage 1 网格一次性构造所有证据和候选，候选先模拟评分、通过硬门后再执行；不在已裁图上不可逆试错。
- 建议二：coverage/weight 主证据定义 reliable mask，像素统计只诊断插值/通道伪影；腐蚀支持域后求 ROI 约束的可靠矩形。
- 建议三：提交裁切事务时同步 SCI/WCS/辅助平面，复测失败自动回退到上一已接受候选，再决定 degraded/manual-review。

#### 几何掩膜与可靠掩膜 (`geometric_mask_vs_reliable_mask`)

- 当前没有显式 geometric mask；场旋黑角只通过 5×5 patch 中是否存在黑像素近似追踪。
- 当前 bad_rows/bad_cols 是亮度、色偏和台阶的混合判定，既不是纯 footprint，也不是带贡献数、噪声和算子支持的 reliable mask。
- 建议 Stage 1 输出 geometric_coverage（至少一帧覆盖）和 coverage_count/weight；Stage 2 再叠加 DQ/rejection、插值支持和下游 halo 得到 reliable/processing-safe mask。

#### 内部空洞策略 (`internal_hole_policy`)

- 当前只从四条外边和四角向内扫描，无法识别或表达画面内部的 drizzle holes、坏列、芯片缝、被拒绝岛和断开区域。
- P1 应保存内部 hard-invalid/low-coverage mask；只有与外边连通且矩形损失可接受的区域由裁切消除。
- ROI 内存在较大孔洞时回退 Stage 1 调整 framing、帧筛选、核或叠加策略；少量坏点继续 mask，展示修补另标 display-only。

### 决策与保护

#### 推荐处理阶段 (`recommended_pipeline_stage`)

- 保留 Stage 2 位于 Stage 1 叠加之后、Stage 3 背景处理之前的顺序；这能在最早阶段隔离黑边对背景、校色和去星的污染。
- Stage 1 应同时产出 framing/coverage 证据，Stage 2 使用它做首个可靠裁切；无需改变 Stage 1-10 的用户可见顺序。
- 由于完整 primary target 在 Stage 3/4 preflight 才冻结，P0 可在 Stage 2 只执行明确硬无效的保守裁切；可能触及弱信号的候选延后到目标复核或标 manual-review。
- Stage 5/6 前只做边界 QA 和工作 halo检查，不再对正式 SCI 进行未记录的二次物理裁切；最终展示可单独生成 display crop。

#### 裁切几何与构图模式 (`crop_geometry_and_framing_mode`)

- 现状只支持四边独立的轴对齐矩形；初检可非对称，能处理一般平移边和部分场旋黑角。
- 它不是严格最大内接矩形：四角循环同时推进相邻边，且四个角依次修改同一 left/right/top/bottom，结果依扫描顺序并可能过裁。
- 没有显式 common/reference/union 模式，也不能为马赛克保留不规则 footprint+mask。
- 建议 P0 保留现有矩形输出但声明 framing=fallback_auto_rect；P1 增加 common可靠矩形、reference+mask、union+mask，旋转构图只在收益显著且允许额外重采样时启用。

#### 验收与回退 (`acceptance_and_fallback`)

- 现状仅 ok/degraded；no-improvement、最终 edge_black 超标、裁切命令失败和保存失败都可能归为 degraded，但原因语义不同。
- 现状即使 degraded 也尽量保存 stage2_corrected.fit，便于流水线继续；但产物没有 science-safe/processing-safe/display-only 能力声明。
- 建议 science-safe 只允许高优先覆盖证据、ROI/辅助平面/WCS 完整；processing-safe 允许后期但不支持定量科学；display-only 允许视觉修补或更紧构图。
- 仅像素 fallback、改善不足或残余坏边归 degraded；ROI冲突、累计损失超预算、证据互相矛盾归 manual-review。
- 回退优先级：撤销未改善候选 → 采用上一个已接受矩形+mask → 回 Stage 1 变更 framing/叠加 → 跳过高风险下游或只导出 review。

#### 误判与失败模式 (`false_positive_and_failure_modes`)

- 真实暗星云、渐晕、宽场背景梯度和负背景会触发近黑/暗边；中心亮星云会放大这种偏差。
- 真实红色发射星云或蓝色反射星云贴边会提高 color_cast，被当作红蓝叠加伪影。
- 整行/列统计会被明亮恒星、衍射结构、卫星轨迹或马赛克面板接缝改变；rolling median 也不能证明异常是无覆盖。
- 全图最大值归一化使阈值受极亮星、饱和核和数据位深影响；不同输入标度下同一绝对阈值不可直接比较。
- NaN/Inf 被归零后无法区分数据错误与预期无覆盖；内部孔洞不在四边扫描范围。
- 四角 5×5逐步算法可能因单个黑点同时推进两边，且各角共享累积边界，未求全局最优矩形。
- 裁后才判断无改善且不回滚会永久丢失数据；多轮在当前图上重新估计中心和百分比使阈值、坐标与总预算漂移。
- 黑边阶段 degraded 时跳过彩边检查，可能留下影响后续的色边；反过来，彩边固定 2.5%也可能过裁真实彩色弱信号。
- stage2 resume 直接信任已有 FITS并从 Stage 3继续，不重跑检测；若旁车、WCS或配置来源缺失，无法证明其满足当前契约。

### 数据完整性与影响

#### 下游影响 (`downstream_impact`)

- 正面：Stage 2 在背景提取前移除明显黑边和彩边，可降低 Stage 3 背景样点、Stage 4校色、Stage 6去星和拉伸被边缘污染的概率。
- 过裁风险：弱信号、背景样区和参考星丢失，Stage 4 platesolve/校色可用星数下降，后续构图不可恢复。
- 漏裁风险：edge_black 或色边会扭曲背景中位数/RMS、被去星或神经网络放大，并在非线性拉伸后更加明显。
- 固定 3px 过紧时，反卷积/去星会重新生成边界伪影；过宽时又造成无证据的信息损失。
- resume 丢失裁切 provenance 会使 Stage 4 报告、质量回溯和同配置复现不完整。
- 没有 mask/weight 时，下游只能把裁后矩形内所有像素当同等可靠，内部孔洞和低覆盖区域无法降权。

#### 质量验证 (`qa_and_validation`)

- P0 单元测试：中心亮/暗、单边黑条、四种旋转黑角、真实红蓝星云贴边、NaN/Inf、整数/浮点不同标度、80px 边界尺寸。
- P0 性质测试：候选坐标始终相对原图可重建；累计裁切不超预算；每次已接受候选嵌套；未改善候选必须回滚。
- P0 报告测试：每轮记录裁前/裁后 edge_black、全局暗比例、各边证据、阈值、总损失、停止原因和状态分类；stage2 resume 可重建同一报告摘要。
- P0 WCS测试：裁前后目标、角点和随机星点 pixel→world→pixel 往返，shape/CRPIX 与累计 left/top 一致；Stage 4重新 platesolve 前后也比较。
- P1 覆盖测试：把已知平移/旋转 footprint 合成 coverage map，验证 common/reference/union、最大可靠矩形、内部孔洞和 guard 腐蚀。
- P1 下游回归：比较裁前后 Stage 3背景、Stage 4解析/校色星数、Stage 5卷积边缘、Stage 6去星、Stage 7拉伸 edge_black 与最终构图。
- 真实数据回归集至少包含暗星云、满幅发射星云、目标贴边、少帧 dither、强场旋、通道错位、马赛克和已叠加外部母版。

#### 裁切溯源 (`crop_provenance`)

- 现有优点：报告记录 original/current/final shape、每次 reason、相对当时图像的 x/y/width/height、removed 四边及相对原图累计 total_crop。
- 缺口：未结构化记录每边判定指标和阈值、候选未采用原因、guard 构成、裁前后完整指标、面积/角视场损失、证据置信度、WCS 摘要和软件版本。
- 缺口：报告与 checkpoint 没有强绑定的 schema/version/hash；resume 不主动加载旁车，无法持续传播 provenance。
- 建议新增 schema、algorithm_version、source_framing、evidence_planes、candidate_rects、accepted_rect、acceptance_class、remaining_risks、WCS/辅助平面校验和及人工覆盖。

### 证据与项目映射

#### 软件与文献实践 (`software_and_literature_practice`)

- Siril 官方已提供 seqapplyreg 的 current/min/max/cog framing；当前项目未显式选择模式，因而把本可在注册几何阶段表达的问题推迟到 RGB亮度扫描。
- Astropy/reproject 返回 footprint，DrizzlePac 输出 SCI/WHT/CTX 并使用 DQ，Montage 输出 area/count；这些实践均证明 coverage/weight 应是主证据。
- DSS 的 Normal/Mosaic/Intersection/Custom Rectangle、APP 的 Full/Reference/Crop、PixInsight 的注册 union/mask/ROI 都把构图策略显式化；当前项目只有一个自动矩形策略。
- Astropy CCDData 切片同步 data、mask 和 WCS，可作为项目未来 crop transaction 的最小数据模型参考。
- 推断：最小风险演进不是立刻重写为复杂最大矩形算法，而是先修复总预算、回滚、配置漂移、WCS/provenance 与输入状态，再逐步把 coverage 引入主判据。

#### 直接证据来源 (`evidence_sources`)

- 项目《pipeline/stages/stage2_view_correction.py》，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/stages/stage2_view_correction.py
- 项目《pipeline/image_metrics.py》，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/image_metrics.py
- 项目《pipeline/models.py》，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/models.py
- 项目《pipeline/stage_support.py》，Stage 1 注册叠加与遗留边裁函数，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/stage_support.py
- 项目《pipeline/stages/stage1_preparation.py》，注册统计，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/stages/stage1_preparation.py
- 项目《pipeline/stages/stage4_color_calibration.py》，裁后几何与 platesolve，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/stages/stage4_color_calibration.py
- 项目《pipeline/seestar_Superimpose.py》，Stage 2续跑与配置边界，本地源码，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/seestar_Superimpose.py
- 项目《seestar_Superimpose_workflow.md》，Stage 2工作流文档，本地文档，访问日期 2026-08-04，/Users/mz/dev/aiseestart/pipeline/seestar_Superimpose_workflow.md
- Siril《Drizzle 与 seqapplyreg framing》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/preprocessing/drizzle.html
- Siril《主界面：CFA 裁切约束》，官方文档，访问日期 2026-08-04，https://siril.readthedocs.io/en/latest/GUI/main-interface.html
- Astropy/reproject《Footprint arrays》，官方文档，更新于 2026-06-25，https://reproject.readthedocs.io/en/stable/footprints.html
- Astropy/reproject《Combining images into mosaics》，官方文档，访问日期 2026-08-04，https://reproject.readthedocs.io/en/stable/mosaicking.html
- Astropy《CCDData Class》，官方文档，访问日期 2026-08-04，https://docs.astropy.org/en/stable/nddata/ccddata.html
- STScI《DrizzlePac Handbook 3.0》，官方手册，2025-01，https://hst-docs.stsci.edu/drizzpac
- DrizzlePac《Single-visit Mosaic Processing》，官方文档，访问日期 2026-08-04，https://drizzlepac.readthedocs.io/en/latest/mast_data_products/svm_processing.html
- Montage《Toolkit Components》，IPAC 官方文档，访问日期 2026-08-04，https://irsa.ipac.caltech.edu/Montage/docs/components.html
- DeepSkyStacker《ResultParameters.cpp》，官方开源实现，访问日期 2026-08-04，https://github.com/deepskystacker/DSS/blob/master/DeepSkyStacker/ResultParameters.cpp
- Astro Pixel Processor《RGB combination and same-size requirements》，第一方管理员答复，2017-08-10 至 2019-10-10，https://www.astropixelprocessor.com/community/faq/rgb-combination-and-same-size-requirements/
- PixInsight《Building large mosaics with StarAlignment》，第一方论坛示例，2010 年，https://pixinsight.com/forum/index.php?threads/building-large-mosaics-with-staralignment.1953/

#### 当前项目适配情况 (`current_project_fit`)

- 优势：Stage 2 位于正确的线性早期；四边可非对称裁切；对严格截黑、暗/亮台阶、红蓝色偏和旋转黑角有多种像素线索；有迭代目标、最大轮数和最小改善门。
- 优势：stage2_corrected.fit 是正式跨运行断点；crop_report 能累计相对原图的 left/top/right/bottom，为进一步 provenance 奠定基础；Stage 4 以裁后尺寸重新解析和计算视场。
- 核心缺口：没有 footprint/DQ/coverage/weight/rejection，像素统计既承担几何有效性又承担视觉伪影判定，无法区分真实暗信号与无数据。
- 安全缺口：没有目标/弱信号 ROI、参考星和背景采样保护；初检 25%扫描、三轮 3.5%补裁和 guard 没有统一总预算。
- 事务缺口：裁后才检查改善，失败不回滚；所有轮次在已裁图重估阈值并改变坐标基准，虽能累计 totals，但不能比较未裁基线上的候选。
- 配置缺口：彩边最大裁切配置未接线，level window 未正式配置，crop_margin 残留但不驱动 Stage 2，实际行为与可调参数不完全一致。
- 几何缺口：角点启发式不是最大可靠矩形，四角顺序会共同改变边界；内部孔洞和不规则马赛克只能漏检。
- 数据契约缺口：只同步裁 SCI，没有 mask/weight/uncertainty；WCS不在 Stage 2验收；resume 不重建 crop_report 和辅助证据。
- 状态缺口：只有 ok/degraded，无法表达 science-safe、processing-safe、display-only 和 manual-review；degraded 原因也未结构化。

#### 实施建议 (`implementation_recommendations`)

- P0 最小改动一：引入相对 original_shape 的统一总裁切预算，覆盖 initial/adaptive/color/guard；报告每边、任一维、面积和估算角视场损失，超限停止并 manual-review/degraded。
- P0 最小改动二：候选先在像素数组或临时 Siril 副本上评估，edge_black 改善不足时不提交或自动回滚；停止原因记录 candidate_rejected_no_improvement。
- P0 最小改动三：让 stage2_color_artifact_max_crop 真正控制彩边预算；把 stage2_level_artifact_window 加入 PipelineConfig、环境映射与范围验证；明确 crop_margin 在 Stage 2废弃，移除或隔离未调用遗留函数。
- P0 最小改动四：报告每边 black_ratio、median、cast、level、使用阈值、guard、候选前后指标和 confidence；将 pixel_only 标为 radiometric_fallback。
- P0 最小改动五：检查原图和裁后图的 finite 比例、数据标度和 WCS；避免把 NaN/Inf 静默归零后丢失类型，按 nonfinite_mask 单独记录。
- P0 最小改动六：为没有目标画像的自动接受加保守门，例如候选总损失大、边缘存在连通弱结构或色偏与结构一致时转 manual-review；实验损失阈值必须可配置并标注非行业标准。
- P0 续跑修复：把 crop report 作为 formal Stage 2 checkpoint 旁车写入 manifest 并带 schema/hash；stage2_corrected_resume 加载它，缺失时明确 provenance_unknown 而不是伪装零裁切。
- P1 中期一：Stage 1 显式设置并记录 seqapplyreg framing/interpolation；原始 Light 路径输出 registered footprint 的 coverage_count/weight，外部母版优先读取已有 mask/weight 扩展。
- P1 中期二：以 coverage/DQ 为主构造 reliable mask，像素近黑/亮度/色偏降级为伪影诊断；从 original grid 一次求 ROI约束最大可靠轴对齐矩形。
- P1 中期三：建立 crop transaction，同步 SCI、mask、coverage/weight、WCS、ROI 和 rejection；任何 shape、CRPIX 或往返校验失败都不提交。
- P1 中期四：在 Stage 3/4目标画像冻结后复核边界候选；冲突时从 stage1_prepared 重算，不在 stage2_corrected 上继续蚕食。
- P1 中期五：guard 由配准残差、插值核、坐标取整和后续最大支持域计算；未知模型 halo 通过 overlap-tile 回归标定。
- P2 长期一：支持 common/reference/union+mask 三种 framing，以及 science-safe、processing-safe、display-only、degraded、manual-review 五级产品。
- P2 长期二：保留内部 holes/坏列/低覆盖 mask，提供同坐标 QA overlay；建立暗星云、满幅星云、场旋、多通道、马赛克和外部母版的金标回归集。
- P2 长期三：版本化阈值和模型，以离线可回归、可打包、可续跑为硬约束；覆盖证据不可用时始终保留当前轻量 fallback。


## 调研文件

- `outline.yaml`：调研范围、条目与执行配置。
- `fields.yaml`：字段定义与不确定项口径。
- `results/*.json`：逐项结构化结果及完整不确定标记。
- `generate_report.py`：本报告的可复现生成脚本。
