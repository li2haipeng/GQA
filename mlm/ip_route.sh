#!/bin/bash

set -xo pipefail

function is_amazon_linux_2() {
    . /etc/os-release
    if [ "$NAME" = "Amazon Linux" ] && [ "$VERSION_ID" = "2" ]; then
        return 0
    else
        return 1
    fi
}

function is_centos_7() {
    . /etc/os-release
    if has_substring "$NAME" "CentOS" && [ "$VERSION_ID" = "7" ]; then
        return 0
    else
        return 1
    fi
}

function is_rhel_7() {
    . /etc/os-release
    if has_substring "$NAME" "Red Hat Enterprise Linux" &&
        echo $VERSION_ID | egrep -q '7\.[0-9]'; then
        return 0
    else
        return 1
    fi
}

function is_suse_15() {
    . /etc/os-release
    if [ "$NAME" = "openSUSE Leap" ]; then
        version_larger_or_equal "$VERSION_ID" "15.3"
        return $?
    elif [ "$NAME" = "SLES" ] && echo $VERSION_ID | egrep -q '15\.[0-9]'; then
        return 0
    else
        return 1
    fi
}

function flush_efa_tables() {
    existing_tables=$(ip route show table all | grep "table" | sed 's/.*\(table.*\)/\1/g' | awk '{print $2}' | grep $EFA_TABLE_PREFIX)
    for tbl in $existing_tables; do
        ip rule del table $tbl 2>/dev/null
        ip route flush table $tbl
    done
}

function get_device_address_with_mask() {
    device=$1
    ip addr show dev $device | grep 'inet ' | head -n 1 | awk '{print $2}'
}

function get_device_address() {
    device=$1
    get_device_address_with_mask $device | cut -d / -f 1
}

function get_device_prefix_length() {
    device=$1
    get_device_address_with_mask $device | cut -d / -f 2
}

function get_device_cidr() {
    device=$1
    # ipcalc is more correct but not all AMIs ship it, e.g. Ubuntus, hence the hack
    addr=$(get_device_address $device)
    prefix_length=$(get_device_prefix_length $device)
    value=$((0xffffffff ^ ((1 << (32 - $prefix_length)) - 1)))
    IFS=. read -r mask1 mask2 mask3 mask4 <<<"$(((value >> 24) & 0xff)).$(((value >> 16) & 0xff)).$(((value >> 8) & 0xff)).$((value & 0xff))"
    IFS=. read -r addr1 addr2 addr3 addr4 <<<$addr
    printf "%d.%d.%d.%d/$prefix_length\n" "$((addr1 & mask1))" "$((addr2 & mask2))" "$((addr3 & mask3))" "$((addr4 & mask4))"
}

function get_device_gateway() {
    device=$1
    if [ $(get_device_prefix_length $device) -eq 32 ]; then
        get_device_address $device
    else
        get_device_cidr $device | sed -E 's/\.[0-9]+\/.+/.1/g'
    fi
}

function get_device_index() {
    device=$1
    # Find the "correct" device index from ip link output
    ip link show dev $device | head -n 1 | cut -d : -f 1
}

function add_device_ip_rule() {
    index=$1
    device=$2
    table_id="${EFA_TABLE_PREFIX}$index"
    addr=$(get_device_address $device)
    gateway=$(get_device_gateway $device)
    # Routing follows lowest metric
    metric=$(($index + 1))
    # proto static overrides dynamic routing
    ip route add default via $gateway proto static dev $device src $addr metric $metric
    ip route add table $table_id default via $gateway proto static dev $device src $addr
    ip rule add from $addr lookup $table_id
}

function delete_default_route() {
    # Delete default routes without metric
    default_route=$(ip route show default | grep -v metric)
    if [ ! -z "$default_route" ]; then
        ip route delete $default_route
    fi
}

function main() {
    if is_amazon_linux_2 || is_centos_7 || is_rhel_7 || is_suse_15; then
        # Use system default routing
        exit 0
    fi

    EFA_TABLE_PREFIX='3834' # 0xEFA

    devices=$(ls -1 /sys/class/net/ | grep -v lo)
    ndevices=$(wc -w <<<$devices)

    if [ $ndevices -gt 1 ]; then
        flush_efa_tables
        for device in $devices; do
            add_device_ip_rule $(get_device_index $device) $device
        done
        delete_default_route
    fi
}

main
