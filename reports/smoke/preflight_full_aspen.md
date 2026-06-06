=== preflight_full_aspen 前置检查报告 ===

【配置摘要】
  case_config_path  : cases/demo_case/pareto_config.yaml
  resolved_path     : E:\Project\GitHub Repo\PAO\cases\demo_case\pareto_config.yaml
  optimizer_type    : pareto_bayesian
  objective_names   : ADN_FLOW, REB_DUTY
  db_path           : E:\Project\GitHub Repo\PAO\cases\demo_case\output\simulation.db
  node_db_path      : E:\Project\GitHub Repo\PAO\cases\demo_case\output\node.db

【检查项目】
  [OK]  配置路径解析：YAML 解析成功，optimizer_type / objective_names / db_path 已填充
  [OK]  Aspen 文件存在性：文件存在
         原始路径: cases/demo_case/二级氢氰化工段.bkp
           解析路径: E:\Project\GitHub Repo\PAO\cases\demo_case\二级氢氰化工段.bkp
  [WW]  优化规模：高风险：预计 80 次 Aspen 评估，不建议作为第一次 full smoke 直接运行。 建议先复制配置并把 n_initial_points/n_iterations 降到 1/1 或 2/1。
         n_initial_points = 20
           n_iterations      = 60
           预计总评估次数    = 80
  [WW]  输出目录风险：检测到已有历史输出，full 运行可能追加或修改数据库
         已存在文件：
             simulation.db (E:\Project\GitHub Repo\PAO\cases\demo_case\output\simulation.db)
             node.db (E:\Project\GitHub Repo\PAO\cases\demo_case\output\node.db)
           建议：复制 case 到 runs/... 临时目录后再运行 full --allow-aspen

【临时运行目录建议】
  建议临时运行目录：runs/demo_case_20260605_170334
  需要复制的内容：
    1. YAML 配置：cases/demo_case/pareto_config.yaml
    2. Aspen .bkp 文件：cases/demo_case/二级氢氰化工段.bkp
    3. 相关语义规则配置（configs/aspen_semantics/）可只读引用
    4. 不要复制 output/ 目录（新运行应从空 output 开始）
    5. 复制后将 n_initial_points 和 n_iterations 降到 2/1 用于首次 smoke
  
  复制后必须在新 YAML 中改写以下路径（否则仍指向原始目录）：
    - simulator.filepath: 改为新目录下的 .bkp 路径（原: cases/demo_case/二级氢氰化工段.bkp）
    - extraction.catalog_db: 改为新 output/ 下的 node.db 路径（原: cases/demo_case/output/node.db）

【综合结论】
  [WARN]  WARN
  不建议直接运行 full --allow-aspen，存在以下警告：
    - 优化规模：高风险：预计 80 次 Aspen 评估，不建议作为第一次 full smoke 直接运行。 建议先复制配置并把 n_initial_points/n_iterations 降到 1/1 或 2/1。
    - 输出目录风险：检测到已有历史输出，full 运行可能追加或修改数据库