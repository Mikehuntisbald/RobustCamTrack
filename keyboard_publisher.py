#!/usr/bin/env python
import rospy

def main():
    rospy.init_node('keyboard_input_node')
    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        input_str = input("请输入 'a' 或 'c' 来修改 flag 参数: ")
        
        if input_str.lower() == 'a' or input_str.lower() == 'c':
            flag_value = input_str.lower() == 'a'
            rospy.set_param('/flag', flag_value)
            print("已将 flag 参数设置为: {}".format(flag_value))
        elif input_str.lower() == 'exit':
            print("退出节点")
            break
        else:
            print("无效输入，请输入 'a' 或 'c'。")

        rate.sleep()

if __name__ == '__main__':
    main()

