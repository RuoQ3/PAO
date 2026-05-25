"""
verify_hap_constants.py — 校验 node.py 中 HAP_NAMES 对应的整数值
是否与当前安装的 Aspen Plus type library（happ.tlb）一致。

用法：
    # 方式 1：通过 EnsureDispatch 启动 Aspen（需要安装 + 许可证）
    python scripts/verify_hap_constants.py

    # 方式 2：直接指定 happ.tlb 路径（无需启动 Aspen，适合 CI 环境）
    python scripts/verify_hap_constants.py --typelib "C:/Program Files/AspenTech/Aspen Plus 10.2/GUI/xeq/happ.tlb"
"""
import argparse
import sys
from pathlib import Path

# 直接从 node.py 导入，避免双份列表漂移
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))
try:
    from src.aspen_driver.node import HAP_NAMES as _HAP_NAMES
except ImportError as _e:
    print(f"ERROR: 无法导入 src.aspen_driver.node.HAP_NAMES：{_e}")
    print("请在项目根目录下运行本脚本：python scripts/verify_hap_constants.py")
    sys.exit(2)

_HAP_NAMES_SET = set(_HAP_NAMES)

PROG_ID = "Apwn.Document"


def _load_via_ensure_dispatch() -> dict[str, int] | None:
    """
    通过 EnsureDispatch 启动 Aspen，填充 gencache，读取常量。
    失败时打印诊断信息并返回 None。
    """
    try:
        import win32com.client
        import win32com.client.gencache
    except ImportError:
        print("ERROR: pywin32 未安装。运行: pip install pywin32")
        return None

    print(f"正在通过 EnsureDispatch('{PROG_ID}') 加载 Aspen type library...")
    try:
        win32com.client.gencache.EnsureDispatch(PROG_ID)
    except Exception as exc:
        _diagnose_dispatch_error(exc)
        return None

    c = win32com.client.constants
    result = {name: int(getattr(c, name)) for name in _HAP_NAMES if hasattr(c, name)}
    missing_after = [n for n in _HAP_NAMES if n not in result]
    if missing_after:
        print(f"注意：EnsureDispatch 成功，但仍有 {len(missing_after)} 个常量未在 constants 中找到。")
        print("可能原因：Aspen Plus 版本不同，或需要重新运行 makepy。")
    return result


def _load_via_typelib(tlb_path: str) -> dict[str, int] | None:
    """
    直接加载指定的 .tlb 文件，不启动 Aspen GUI/服务器。
    适合 CI 环境或无许可证的机器。
    """
    try:
        import pythoncom
    except ImportError:
        print("ERROR: pywin32 未安装。运行: pip install pywin32")
        return None

    print(f"正在从 typelib 文件加载：{tlb_path}")
    try:
        tlib = pythoncom.LoadTypeLib(tlb_path)
    except Exception as exc:
        print(f"ERROR: 无法加载 typelib：{exc}")
        print("请确认路径正确，且文件存在。")
        return None

    # 遍历 typelib 中所有枚举，收集 HAP_ 常量
    result: dict[str, int] = {}
    for i in range(tlib.GetTypeInfoCount()):
        try:
            typeinfo = tlib.GetTypeInfo(i)
            attr = typeinfo.GetTypeAttr()
            import pythoncom as _pc
            if attr.typekind != _pc.TKIND_ENUM:
                continue
            for j in range(attr.cVars):
                vardesc = typeinfo.GetVarDesc(j)
                names = typeinfo.GetNames(vardesc.memid, 1)
                if names and names[0] in _HAP_NAMES_SET:
                    result[names[0]] = int(vardesc.value)
        except Exception:
            continue

    return result


def _diagnose_dispatch_error(exc: Exception) -> None:
    msg = str(exc).lower()
    print(f"\nERROR: EnsureDispatch 失败：{exc}\n")
    if "class not registered" in msg or "0x80040154" in msg:
        print("诊断：Aspen Plus COM 服务器未注册。")
        print("  - 确认 Aspen Plus 已正确安装。")
        print("  - 尝试以管理员身份运行：regsvr32 apwn.exe")
    elif "license" in msg or "0x80070005" in msg or "access" in msg:
        print("诊断：许可证或权限问题。")
        print("  - 确认 Aspen Plus 许可证有效且当前用户有权访问。")
    elif "server execution failed" in msg or "0x80080005" in msg or "服务器运行失败" in msg:
        print("诊断：Aspen Plus 服务器启动失败（可能超时或许可证不可用）。")
        print("  - 尝试手动启动 Aspen Plus，确认可以正常打开。")
        print("  - 或使用 --typelib 参数直接指定 happ.tlb 路径，跳过服务器启动：")
        print('    python scripts/verify_hap_constants.py --typelib "C:/...path.../happ.tlb"')
    elif "makepy" in msg or "gencache" in msg:
        print("诊断：gencache 生成失败（可能是只读文件系统或权限问题）。")
        print("  - 尝试手动运行：python -m win32com.client.makepy")
    else:
        print("诊断：未知错误。请确认：")
        print("  1. Aspen Plus 已安装并可正常启动。")
        print("  2. 当前 Python 环境已安装 pywin32（pip install pywin32）。")
        print("  3. 以管理员身份运行本脚本。")
        print("  4. 或使用 --typelib 参数直接指定 happ.tlb 路径。")


def _report(loaded: dict[str, int]) -> int:
    found = set(loaded)
    missing = [n for n in _HAP_NAMES if n not in found]

    # 分组：节点属性 vs COMPSTATUS 位掩码
    attr_names = [n for n in _HAP_NAMES if not n.startswith("HAP_RESULTS_") and n != "HAP_NORESULTS"]
    mask_names = [n for n in _HAP_NAMES if n.startswith("HAP_RESULTS_") or n == "HAP_NORESULTS"]

    print(f"\n=== HAPAttributeNumber 校验结果 ===")
    print(f"目标常量  : {len(_HAP_NAMES)}  （节点属性 {len(attr_names)} + 位掩码 {len(mask_names)}）")
    print(f"找到常量  : {len(found)}")
    print(f"缺失常量  : {len(missing)}")

    if found:
        print("\n[节点属性常量]")
        print(f"  {'常量名':<24} {'值':>8}")
        print(f"  {'-'*24} {'-'*8}")
        for name in attr_names:
            if name in loaded:
                print(f"  {name:<24} {loaded[name]:>8}")
            else:
                print(f"  {name:<24} {'MISSING':>8}  <--")

        print("\n[COMPSTATUS 位掩码常量]")
        print(f"  {'常量名':<28} {'值':>8}  {'十六进制':>10}")
        print(f"  {'-'*28} {'-'*8}  {'-'*10}")
        for name in mask_names:
            if name in loaded:
                v = loaded[name]
                print(f"  {name:<28} {v:>8}  {v:#010x}")
            else:
                print(f"  {name:<28} {'MISSING':>8}  {'':>10}  <--")

    if missing:
        print(f"\n[MISSING] 以下 {len(missing)} 个常量未在 type library 中找到：")
        for name in missing:
            print(f"  {name}")
        print("可能原因：Aspen Plus 版本不同，或该常量名在此版本中已更名。")
        return 1

    print("\n所有常量均已找到，node.py 的 hap_constants 加载路径可正常工作。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="校验 Aspen Plus HAPAttributeNumber 常量")
    parser.add_argument(
        "--typelib",
        metavar="PATH",
        help="直接指定 happ.tlb 路径（不启动 Aspen，适合 CI 环境）",
    )
    args = parser.parse_args()

    if args.typelib:
        loaded = _load_via_typelib(args.typelib)
    else:
        loaded = _load_via_ensure_dispatch()

    if loaded is None:
        return 2

    return _report(loaded)


if __name__ == "__main__":
    sys.exit(main())
