from arclet.alconna import Alconna, Args, Subcommand
from nonebot_plugin_alconna import on_alconna, Match
from nonebot_plugin_uninfo import Uninfo

from .func.easy_ban import (
    add_banned_user,
    remove_banned_user,
    add_banned_group,
    remove_banned_group,
    get_banned_users,
    get_banned_groups,
)
from ..utils import APP_CONFIG, extract_session_info


async def _perm(session: Uninfo) -> bool:
    return session.user.id in APP_CONFIG.admin_users


easyban_alc = Alconna(
    "easyban",
    Subcommand(
        "user",
        Subcommand("add", Args["user_id?", str], help_text="将用户加入限制列表"),
        Subcommand("remove", Args["user_id?", str], help_text="将用户从限制列表中移除"),
        Subcommand("list", help_text="查看当前在限制列表中的用户"),
        help_text="用户限制管理",
    ),
    Subcommand(
        "group",
        Subcommand("add", Args["group_id?", str], help_text="将群组加入限制列表"),
        Subcommand(
            "remove", Args["group_id?", str], help_text="将群组从限制列表中移除"
        ),
        Subcommand("list", help_text="查看当前在限制列表中的群组"),
        help_text="群组限制管理",
    ),
)

easyban_cmd = on_alconna(
    easyban_alc, aliases={"eb", "ban"}, use_cmd_start=True, block=True, rule=_perm
)


@easyban_cmd.assign("user.add")
async def handle_user_add(user_id: Match[str], session: Uninfo):
    _user = user_id.result if user_id.available else None

    if not _user:
        await easyban_cmd.finish("请提供需要限制的用户ID哦~")

    if _user == session.user.id:
        await easyban_cmd.finish("自己可不能限制自己呢~")

    if _user in APP_CONFIG.admin_users:
        await easyban_cmd.finish("管理员可是不能被限制的哦~")

    await add_banned_user(_user)
    await easyban_cmd.finish(f"用户 {_user} 已被加入限制列表~")


@easyban_cmd.assign("user.remove")
async def handle_user_remove(user_id: Match[str]):
    _user = user_id.result if user_id.available else None

    if not _user:
        await easyban_cmd.finish("请提供需要解除限制的用户ID哦~")

    await remove_banned_user(_user)
    await easyban_cmd.finish(f"用户 {_user} 已被成功解除限制啦~")


@easyban_cmd.assign("user.list")
async def handle_user_list():
    users = await get_banned_users()
    if not users:
        await easyban_cmd.finish("当前没有任何用户在限制列表中哦~")

    user_list = "\n".join(sorted(users))
    await easyban_cmd.finish(f"以下是当前在限制列表中的用户：\n{user_list}")


@easyban_cmd.assign("group.add")
async def handle_group_add(group_id: Match[str], session: Uninfo):
    _group = group_id.result if group_id.available else None
    if _group is None:
        info = extract_session_info(session)
        _group = info["group_id"]

    if not _group:
        await easyban_cmd.finish("请提供需要限制的群组ID，或者在群组中使用该命令哦~")

    await add_banned_group(_group)
    await easyban_cmd.finish(f"群组 {_group} 已被加入限制列表~")


@easyban_cmd.assign("group.remove")
async def handle_group_remove(group_id: Match[str], session: Uninfo):
    _group = group_id.result if group_id.available else None
    if _group is None:
        info = extract_session_info(session)
        _group = info["group_id"]

    if not _group:
        await easyban_cmd.finish(
            "请提供需要解除限制的群组ID，或者在群组中使用该命令哦~"
        )

    await remove_banned_group(_group)
    await easyban_cmd.finish(f"群组 {_group} 已被成功解除限制啦~")


@easyban_cmd.assign("group.list")
async def handle_group_list():
    groups = await get_banned_groups()
    if not groups:
        await easyban_cmd.finish("当前没有任何群组在限制列表中哦~")

    group_list = "\n".join(sorted(groups))
    await easyban_cmd.finish(f"以下是当前在限制列表中的群组：\n{group_list}")
