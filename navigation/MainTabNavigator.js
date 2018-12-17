import React from 'react';
import { Platform } from 'react-native';
import { createStackNavigator, createBottomTabNavigator } from 'react-navigation';

import TabBarIcon from '../components/TabBarIcon';
import HomeScreen from '../screens/HomeScreen';
import NewDocumentScreen from '../screens/NewDocumentScreen';
import SettingsScreen from '../screens/SettingsScreen';
import DocumentScreen from '../screens/DocumentScreen';
import DocumentsScreen from '../screens/DocumentsScreen';
import NewQuizScreen from '../screens/NewQuizScreen';
import TestScreen from '../screens/TestScreen';

const HomeStack = createStackNavigator({
  Home: HomeScreen,
  Doc: DocumentScreen,
  OldQuiz: NewQuizScreen,
  Docs: DocumentsScreen,
  Test: TestScreen,
}, {
  initialRouteName: 'Home',
});

HomeStack.navigationOptions = {
  tabBarLabel: 'Home',
  tabBarIcon: ({ focused }) => (
    <TabBarIcon
      focused={focused}
      name={
        Platform.OS === 'ios'
          ? `ios-information-circle${focused ? '' : '-outline'}`
          : 'md-information-circle'
      }
    />
  ),
};

const NewDocStack = createStackNavigator({
  NewDoc: NewDocumentScreen,
  NewQuiz: NewQuizScreen,
}, {
  initialRouteName: 'NewDoc',
});

NewDocStack.navigationOptions = {
  tabBarLabel: 'New Quiz',
  tabBarIcon: ({ focused }) => (
    <TabBarIcon
      focused={focused}
      name={Platform.OS === 'ios' ? 'ios-link' : 'md-link'}
    />
  ),
};

// const SettingsStack = createStackNavigator({
//   Settings: SettingsScreen,
// });
//
// SettingsStack.navigationOptions = {
//   tabBarLabel: 'Settings',
//   tabBarIcon: ({ focused }) => (
//     <TabBarIcon
//       focused={focused}
//       name={Platform.OS === 'ios' ? 'ios-options' : 'md-options'}
//     />
//   ),
// };

export default createBottomTabNavigator({
  HomeStack,
  NewDocStack,
  // SettingsStack,
});
