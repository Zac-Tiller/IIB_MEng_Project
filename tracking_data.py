import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from mplsoccer import Pitch


def load_tracking_data():
    tracking_df = pd.read_csv(
        '/Users/zactiller/Documents/IIB_MEng_Project/Tracking_Data/Sample_Game_1_RawTrackingData_Home_Team.csv',
        skiprows=2)
    tracking_df.columns = [['Period', 'Frame', 'Time [s]',
                            'P11 X', 'P11 Y',
                            'P1 X', 'P1 Y',
                            'P2 X', 'P2 Y',
                            'P3 X', 'P3 Y',
                            'P4 X', 'P4 Y',
                            'P5 X', 'P5 Y',
                            'P6 X', 'P6 Y',
                            'P7 X', 'P7 Y',
                            'P8X', 'P8 Y',
                            'P9 X', 'P9 Y',
                            'P10 X', 'P10 Y',
                            'P12X', 'P12 Y',
                            'P13X', 'P13 Y',
                            'P14X', 'P14 Y',
                            'Ball X', 'Ball Y']]

    return tracking_df
#
# def show_player_path(tracking_df, player_number, start, end):
#     start_min = int(start[:2])
#     start_sec = int(start[3:])
#
#     end_min = int(end[:2])
#     end_sec = int(end[3:])
#
#     i_start = int((start_min*60 + start_sec)/0.04)
#     i_end = int((end_min*60 + end_sec)/0.04)
#
#     times = tracking_df['Time [s]'].values[i_start: i_end]
#     colors = (times - times[0]) / (times - times[0])[-1]
#     colors = times/60
#
#     # # Create scatter plot with time-based color mapping
#     # fig, ax = plt.subplots()
#     # scatter = ax.scatter(tracking_df['P{} X'.format(player_number)].values[i_start:i_end], tracking_df['P{} Y'.format(player_number)].values[i_start:i_end], c=colors.flatten(), cmap='RdYlGn_r',
#     #                      s=0.25)
#     #
#     # # Set color bar label
#     # cbar = plt.colorbar(scatter)
#     #
#     # # Add a colorbar to show the time-color mapping
#     # cbar.set_label('Time (Min.Sec)')
#     #
#     # # Add pitch boundaries and labels
#     # plt.xlim(0, 1)
#     # plt.ylim(0, 1)
#     # ax.set_xlabel('X position')
#     # ax.set_ylabel('Y position')
#     # ax.set_title('Player {} Position Over Time'.format(player_number))
#     # plt.show()
#
#     # Define pitch dimensions
#     pitch_length = 105.0  # in meters
#     pitch_width = 68.0  # in meters
#
#     # Create scatter plot with time-based color mapping
#     fig, ax = plt.subplots(figsize=(10, 6))
#     scatter = ax.scatter(tracking_df['P{} X'.format(player_number)].values[i_start:i_end],
#                          tracking_df['P{} Y'.format(player_number)].values[i_start:i_end], c=colors.flatten(),
#                          cmap='RdYlGn_r', s=100)
#
#     # Set color bar label
#     cbar = plt.colorbar(scatter)
#     cbar.set_label('Time (Min.Sec)')
#
#     # Add pitch boundaries and labels
#     plt.xlim(0, pitch_length)
#     plt.ylim(0, pitch_width)
#     ax.set_xlabel('X position (m)')
#     ax.set_ylabel('Y position (m)')
#     ax.set_title('Player {} Position Over Time'.format(player_number))
#
#     # Add pitch lines and markings
#     plt.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color='white')
#     plt.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color='white')
#     plt.plot([0, pitch_length], [pitch_width / 2, pitch_width / 2], color='white')
#     plt.plot([9.15, 9.15], [0, pitch_width], color='white', linewidth=2)
#     plt.plot([pitch_length - 9.15, pitch_length - 9.15], [0, pitch_width], color='white', linewidth=2)
#     circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color='white', fill=False, linewidth=2)
#     ax.add_artist(circle)
#
#     # Add a faint light green background for grass
#     ax.set_facecolor('#E5F2E5')
#
#     plt.show()
#
#     return tracking_df['P{} X'.format(player_number)].values[i_start:i_end], tracking_df['P{} Y'.format(player_number)].values[i_start:i_end], times

def show_player_path(tracking_df, sub_sample_rate, player_number, start, end):
    start_min = int(start[:2])
    start_sec = int(start[3:])

    end_min = int(end[:2])
    end_sec = int(end[3:])

    i_start = int((start_min * 60 + start_sec) / 0.04)
    i_end = int((end_min * 60 + end_sec) / 0.04)

    times = tracking_df['Time [s]'].values[i_start:i_end][::sub_sample_rate]
    colors = (times - times[0]) / (times - times[0])[-1][::sub_sample_rate]
    colors = times / 60

    # Define pitch dimensions
    pitch_length = 105.0  # in meters
    pitch_width = 68.0  # in meters

    # Create scatter plot with time-based color mapping
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a soccer pitch
    pitch = Pitch(pitch_type='statsbomb', line_color='white',
                  pitch_color='#E5F2E5')

    # Draw the soccer pitch
    pitch.draw(ax=ax)

    scatter = ax.scatter(
        tracking_df['P{} X'.format(player_number)].values[i_start:i_end][::sub_sample_rate] * pitch_length,
        tracking_df['P{} Y'.format(player_number)].values[i_start:i_end][::sub_sample_rate] * pitch_width,
        c=colors.flatten(), cmap='RdYlGn_r', s=10)

    # Set color bar label
    cbar = plt.colorbar(scatter)
    cbar.set_label('Time (Min.Sec)')


    # Create scatter plot with time-based color mapping
    # scatter = ax.scatter(player_x, player_y, c=colors.flatten(), cmap='RdYlGn_r', s=100)

    # # Set color bar label
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Time (Min.Sec)')

    # Set pitch boundaries and labels
    # ax.set_xlim(0, pitch_length)
    # ax.set_ylim(0, pitch_width)
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_title('Player {} Position Over Time'.format(player_number))

    # Show the plot
    plt.show()

    # return player_x, player_y, times

    # # Add pitch boundaries and labels
    # plt.xlim(0, pitch_length)
    # plt.ylim(0, pitch_width)
    # ax.set_xlabel('X position (m)')
    # ax.set_ylabel('Y position (m)')
    # ax.set_title('Player {} Position Over Time'.format(player_number))
    #
    # # Add pitch lines and markings
    # plt.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color='white')
    # plt.plot([pitch_length / 2, pitch_length / 2], [0, pitch_width], color='white')
    # plt.plot([9.15, 9.15], [0, pitch_width], color='white', linewidth=2)
    # plt.plot([pitch_length - 9.15, pitch_length - 9.15], [0, pitch_width], color='white', linewidth=2)
    # circle = plt.Circle((pitch_length / 2, pitch_width / 2), 9.15, color='white', fill=False, linewidth=2)
    # ax.add_artist(circle)
    #
    # # # Draw boxes for opponents
    # # opponent_boxes = tracking_df[['Opponent{}_X'.format(i) for i in range(1, 12)]].values[i_start:i_end] * pitch_length
    # # opponent_boxes_y = tracking_df[['Opponent{}_Y'.format(i) for i in range(1, 12)]].values[i_start:i_end] * pitch_width
    # # for box_x, box_y in zip(opponent_boxes, opponent_boxes_y):
    # #     rect = plt.Rectangle((box_x - 0.5, box_y - 0.5), 1.0, 1.0, linewidth=1, edgecolor='white', facecolor='none')
    # #     ax.add_patch(rect)
    #
    # # Draw boxes for goals
    # goal_boxes = [[0, pitch_width / 2 - 3.66], [0, pitch_width / 2 + 2.66], [pitch_length, pitch_width / 2 - 3.66],
    #               [pitch_length, pitch_width / 2 + 2.66]]
    # for box_x, box_y in goal_boxes:
    #     rect = plt.Rectangle((box_x - 0.5, box_y - 0.5), 1.0, 1.0, linewidth=1, edgecolor='white', facecolor='none')
    #     ax.add_patch(rect)
    #
    # # Draw rectangles around the goals
    # goal_width = 7.32  # in meters
    # goal_height = 2.44  # in meters
    # goal_area_height = 5.5  # in meters (height of the D-like shape)
    #
    # # Left goal
    # left_goal_x = 0.0
    # left_goal_y = (pitch_width - goal_height) / 2
    # left_goal_rect = patches.Rectangle((left_goal_x, left_goal_y), goal_width, goal_height, linewidth=1,
    #                                    edgecolor='white',
    #                                    facecolor='none')
    # ax.add_patch(left_goal_rect)
    #
    # # Right goal
    # right_goal_x = pitch_length - goal_width
    # right_goal_y = (pitch_width - goal_height) / 2
    # right_goal_rect = patches.Rectangle((right_goal_x, right_goal_y), goal_width, goal_height, linewidth=1,
    #                                     edgecolor='white',
    #                                     facecolor='none')
    # ax.add_patch(right_goal_rect)
    #
    # # Draw the D-like shape on the outside of the left goal
    # d_shape_radius = 9.15  # in meters
    # d_shape_center_x = d_shape_radius
    # d_shape_center_y = pitch_width / 2
    # d_shape = patches.Circle((d_shape_center_x, d_shape_center_y), d_shape_radius, color='white', fill=False,
    #                          linewidth=1)
    # ax.add_patch(d_shape)
    # d_shape_arc = patches.Arc((d_shape_center_x, d_shape_center_y), 2 * d_shape_radius, 2 * d_shape_radius, theta1=20,
    #                           theta2=160, color='white', linewidth=1)
    # ax.add_patch(d_shape_arc)
    #
    # # Draw the D-like shape on the outside of the right goal
    # d_shape_center_x = pitch_length - d_shape_radius
    # d_shape = patches.Circle((d_shape_center_x, d_shape_center_y), d_shape_radius, color='white', fill=False,
    #                          linewidth=1)
    # ax.add_patch(d_shape)
    # d_shape_arc = patches.Arc((d_shape_center_x, d_shape_center_y), 2 * d_shape_radius, 2 * d_shape_radius, theta1=-160,
    #                           theta2=-20, color='white', linewidth=1)
    # ax.add_patch(d_shape_arc)
    #
    # # Draw the '6-yard box' for each goal
    # box_width = 6.0  # in meters
    # box_height = 4.0  # in meters
    # box_y = (pitch_width - box_height) / 2
    #
    # # Left goal '6-yard box'
    # left_box_x = 0.0
    # left_box_rect = patches.Rectangle((left_box_x, box_y), box_width, box_height, linewidth=1, edgecolor='white',
    #                                   facecolor='none', linestyle='dashed')
    # ax.add_patch(left_box_rect)
    #
    # # Right goal '6-yard box'
    # right_box_x = pitch_length - box_width
    # right_box_rect = patches.Rectangle((right_box_x, box_y), box_width, box_height, linewidth=1, edgecolor='white',
    #                                    facecolor='none', linestyle='dashed')
    # ax.add_patch(right_box_rect)
    #
    #
    # # Add a faint light green background for grass
    # ax.set_facecolor('#E5F2E5')
    #
    # plt.show()

    return (
        tracking_df['P{} X'.format(player_number)].values[i_start:i_end][::sub_sample_rate] * pitch_length,
        tracking_df['P{} Y'.format(player_number)].values[i_start:i_end][::sub_sample_rate] * pitch_width,
        tracking_df['Time [s]'].values[i_start:i_end][::sub_sample_rate], fig, ax
    )


if __name__ == '__main__':
    tracking_df = load_tracking_data()
    player_x, player_y, times, fig, ax = show_player_path(tracking_df, 10, start='18:30', end='18:45')



#
# if __name__=='__main()__':
#
#     tracking_df = load_tracking_data()
#
#     player_x, player_y, times = show_player_path(tracking_df, 10, start='18:30', end='18:45')
#
#
#



